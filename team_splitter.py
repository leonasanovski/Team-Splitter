"""
Team Splitter: splits the players coming to the next match into two teams that are as even
as possible, using the live Zoki Liga database.

Each player's strength comes from two things the league app already records:
  1. Results: a TrueSkill rating that is replayed over every match in date order, so a
     player's rating rises when their teams do better than expected and falls when they
     do worse.
  2. Goal contributions: goals + assists - own goals per match, shrunk towards the league
     average for players with few matches.
Both are turned into z-scores, mixed with STRENGTH_WEIGHTS and put on a 50 + 10z scale.

Goalkeepers: when two of the selected players are goalkeepers by main position, they go to
opposite teams. Otherwise the teams are split purely on strength and everyone rotates in goal.

Usage:
  python team_splitter.py split Хито Маки Марио ...   (nicknames or player ids)
  python team_splitter.py ratings                    (every player's current strength)
  python team_splitter.py backtest                   (how well strength predicted past results)

Database connection uses the same variables as the Zoki Liga backend:
DB_HOST, DB_PORT, DB_NAME, DB_USERNAME, DB_PASSWORD.
"""

import argparse
import itertools
import math
import os
import sys
from dataclasses import dataclass, field
from statistics import NormalDist

import psycopg

# ---------- TrueSkill parameters ----------
INITIAL_MU = 25.0
INITIAL_SIGMA = INITIAL_MU / 3
BETA = 20.0                 # performance noise per player; large because 6v6 results are noisy (best log loss in backtest)
TAU = INITIAL_SIGMA / 100   # uncertainty added before every match, so ratings can follow form
DRAW_PROBABILITY = 0.14     # share of drawn matches in seasons 2023/24 - 2025/26

# ---------- Strength formula ----------
STRENGTH_WEIGHTS = {
    'results': 0.5,             # TrueSkill mu
    'goal_contributions': 0.5,  # shrunk (goals + assists - own goals) per match
}
SHRINKAGE_MATCHES = 10      # a player's per-match rate counts as this many matches of league average
NEW_PLAYER_STRENGTH = 50.0  # someone with no matches yet is treated as average
BACKTEST_WARMUP_MATCHES = 40

GOALKEEPER = 'GOALKEEPER'
NORMAL = NormalDist()


@dataclass
class Player:
    id: int
    nickname: str
    main_position: str
    is_active: bool
    mu: float = INITIAL_MU
    sigma: float = INITIAL_SIGMA
    matches: int = 0
    goal_contributions: int = 0
    strength: float = NEW_PLAYER_STRENGTH


@dataclass
class Match:
    id: int
    team_1_score: int
    team_2_score: int
    team_1: list = field(default_factory=list)  # [(player_id, goals + assists - own_goals)]
    team_2: list = field(default_factory=list)


# ---------- Data ----------

def connect():
    return psycopg.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        dbname=os.getenv('DB_NAME', 'zoki_liga'),
        user=os.getenv('DB_USERNAME', 'postgres'),
        password=os.getenv('DB_PASSWORD'),
    )


def load_league(connection):
    players = {
        player_id: Player(player_id, nickname, main_position, is_active)
        for player_id, nickname, main_position, is_active in connection.execute(
            'SELECT id, nickname, main_position, is_active FROM core.player')
    }
    matches = {
        match_id: Match(match_id, team_1_score, team_2_score)
        for match_id, team_1_score, team_2_score in connection.execute(
            'SELECT id, team_1_score, team_2_score FROM core."match" ORDER BY date_time_played, id')
    }
    for match_id, player_id, team, goals, assists, own_goals in connection.execute(
            'SELECT match_id, player_id, team, goals, assists, own_goals FROM core.match_player_record'):
        side = matches[match_id].team_1 if team == 'TEAM_1' else matches[match_id].team_2
        side.append((player_id, goals + assists - own_goals))
    # dicts keep insertion order, so this is chronological
    return players, [match for match in matches.values() if match.team_1 and match.team_2]


# ---------- TrueSkill (two teams, with draws) ----------

def win_probability(team_1_mus, team_2_mus, sigmas):
    spread = math.sqrt(sum(s * s for s in sigmas) + len(sigmas) * BETA * BETA)
    return NORMAL.cdf((sum(team_1_mus) - sum(team_2_mus)) / spread)


def update_ratings(players, match):
    for player_id, _ in match.team_1 + match.team_2:
        players[player_id].sigma = math.hypot(players[player_id].sigma, TAU)

    team_1 = [players[player_id] for player_id, _ in match.team_1]
    team_2 = [players[player_id] for player_id, _ in match.team_2]
    everyone = team_1 + team_2
    spread = math.sqrt(sum(p.sigma ** 2 for p in everyone) + len(everyone) * BETA * BETA)
    draw_margin = NORMAL.inv_cdf((DRAW_PROBABILITY + 1) / 2) * math.sqrt(len(everyone)) * BETA / spread
    difference = (sum(p.mu for p in team_1) - sum(p.mu for p in team_2)) / spread

    if match.team_1_score == match.team_2_score:
        t, e = difference, draw_margin
        denominator = max(NORMAL.cdf(e - t) - NORMAL.cdf(-e - t), 1e-12)
        v = (NORMAL.pdf(-e - t) - NORMAL.pdf(e - t)) / denominator
        w = v * v + ((e - t) * NORMAL.pdf(e - t) + (e + t) * NORMAL.pdf(e + t)) / denominator
        team_1_sign = 1
    else:
        team_1_sign = 1 if match.team_1_score > match.team_2_score else -1
        t = team_1_sign * difference - draw_margin
        v = NORMAL.pdf(t) / max(NORMAL.cdf(t), 1e-12)
        w = v * (v + t)

    for team, sign in ((team_1, team_1_sign), (team_2, -team_1_sign)):
        for player in team:
            variance_share = player.sigma ** 2 / spread
            player.mu += sign * variance_share * v
            player.sigma *= math.sqrt(max(1 - variance_share / spread * w, 1e-4))


def record_appearances(players, match):
    for player_id, contribution in match.team_1 + match.team_2:
        players[player_id].matches += 1
        players[player_id].goal_contributions += contribution


# ---------- Strength ----------

def z_scores(values):
    mean = sum(values) / len(values)
    deviation = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1e-9
    return [(v - mean) / deviation for v in values]


def compute_strengths(players):
    rated = [p for p in players.values() if p.matches > 0]
    if not rated:
        return
    league_rate = sum(p.goal_contributions for p in rated) / sum(p.matches for p in rated)
    shrunk_rates = [(p.goal_contributions + SHRINKAGE_MATCHES * league_rate) / (p.matches + SHRINKAGE_MATCHES)
                    for p in rated]
    for player, results_z, contributions_z in zip(rated, z_scores([p.mu for p in rated]), z_scores(shrunk_rates)):
        player.strength = 50 + 10 * (STRENGTH_WEIGHTS['results'] * results_z
                                     + STRENGTH_WEIGHTS['goal_contributions'] * contributions_z)


def build_ratings(players, matches):
    for match in matches:
        update_ratings(players, match)
        record_appearances(players, match)
    compute_strengths(players)


# ---------- Splitting ----------

def goalkeepers_split_evenly(team, goalkeepers):
    if len(goalkeepers) != 2:
        return True
    return sum(p in team for p in goalkeepers) == 1


def best_splits(selected, how_many=3):
    """Every split with team sizes differing by at most one, ranked by strength gap."""
    goalkeepers = [p for p in selected if p.main_position == GOALKEEPER]
    first, rest = selected[0], selected[1:]
    size = len(selected) // 2
    candidates = []
    # fixing the first player on team A means each split is seen once, not twice
    for team_a_size in {size, len(selected) - size}:
        for others in itertools.combinations(rest, team_a_size - 1):
            team_a = [first, *others]
            if not goalkeepers_split_evenly(team_a, goalkeepers):
                continue
            team_b = [p for p in rest if p not in others]
            gap = abs(sum(p.strength for p in team_a) - sum(p.strength for p in team_b))
            candidates.append((gap, team_a, team_b))
    candidates.sort(key=lambda candidate: candidate[0])
    return candidates[:how_many]


def find_players(players, names):
    by_nickname = {p.nickname.strip().lower(): p for p in players.values()}
    found, unknown = [], []
    for name in names:
        player = players.get(int(name)) if name.isdigit() else by_nickname.get(name.strip().lower())
        (found.append(player) if player else unknown.append(name))
    if unknown:
        sys.exit(f'Unknown players: {", ".join(unknown)}')
    if len(set(p.id for p in found)) != len(found):
        sys.exit('A player is listed twice.')
    if len(found) < 4:
        sys.exit('Need at least 4 players to split.')
    return found


# ---------- Commands ----------

def print_team(label, team):
    total = sum(p.strength for p in team)
    print(f'  {label} (strength {total:.1f})')
    for p in sorted(team, key=lambda p: -p.strength):
        goalkeeper = '  [GK]' if p.main_position == GOALKEEPER else ''
        print(f'    {p.nickname:<16} {p.strength:5.1f}{goalkeeper}')


def command_split(players, matches, names):
    build_ratings(players, matches)
    selected = find_players(players, names)
    for rank, (gap, team_a, team_b) in enumerate(best_splits(selected), start=1):
        sigmas = [p.sigma for p in team_a + team_b]
        chance = win_probability([p.mu for p in team_a], [p.mu for p in team_b], sigmas)
        print(f'\nOption {rank}: strength gap {gap:.1f}, results model gives team A {chance:.0%}')
        print_team('Team A', team_a)
        print_team('Team B', team_b)


def command_ratings(players, matches):
    build_ratings(players, matches)
    print(f'{"player":<16} {"position":<11} {"matches":>7} {"strength":>8} {"mu":>6} {"sigma":>6}')
    for p in sorted(players.values(), key=lambda p: -p.strength):
        if p.matches:
            print(f'{p.nickname:<16} {p.main_position:<11} {p.matches:>7} {p.strength:8.1f} {p.mu:6.1f} {p.sigma:6.1f}')


def command_backtest(players, matches):
    """Predict each match before learning from it, after a warm-up season."""
    picked, decisive = 0, 0
    for index, match in enumerate(matches):
        if index >= BACKTEST_WARMUP_MATCHES and match.team_1_score != match.team_2_score:
            compute_strengths(players)
            team_1 = sum(players[pid].strength for pid, _ in match.team_1)
            team_2 = sum(players[pid].strength for pid, _ in match.team_2)
            if team_1 != team_2:
                decisive += 1
                picked += (team_1 > team_2) == (match.team_1_score > match.team_2_score)
        update_ratings(players, match)
        record_appearances(players, match)
    print(f'Stronger team won {picked} of {decisive} decisive matches '
          f'({picked / max(decisive, 1):.0%}) after the first {BACKTEST_WARMUP_MATCHES} matches.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('split', help='split the given players into two teams').add_argument('players', nargs='+')
    commands.add_parser('ratings', help='list every player\'s current strength')
    commands.add_parser('backtest', help='check how well strength predicted past matches')
    arguments = parser.parse_args()

    with connect() as connection:
        players, matches = load_league(connection)
    if arguments.command == 'split':
        command_split(players, matches, arguments.players)
    elif arguments.command == 'ratings':
        command_ratings(players, matches)
    else:
        command_backtest(players, matches)


if __name__ == '__main__':
    main()
