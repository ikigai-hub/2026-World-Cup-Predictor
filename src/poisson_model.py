import numpy as np
import pandas as pd
from scipy.stats import poisson

def compute_ratings(df):
    df['goal_total'] = df['goal1'] + df['goal2']
    avg = df['goal_total'].sum() / len(df['goal_total'])

    unique_teams = pd.concat([df['team1'], df['team2']]).unique().tolist()

    goals = {}
    conceded = {}
    for team in unique_teams:
        scored_as_team1 = df[df['team1'] == team]['goal1']
        conceded_as_team1 = df[df['team1'] == team]['goal2']
        scored_as_team2 = df[df['team2'] == team]['goal2']
        conceded_as_team2 = df[df['team2'] == team]['goal1']
        
        goals_scored = pd.concat([scored_as_team1, scored_as_team2])
        goals_conceded = pd.concat([conceded_as_team1, conceded_as_team2])
        
        n = len(goals_scored)
        k = 3
        smoothed_attack = (goals_scored.mean() * n + 1.0 * k) / (n + 3)
        smoothed_defence = (goals_conceded.mean() * n + 1.0 * k) / (n + 3)

        goals[team] = smoothed_attack / avg
        conceded[team] = smoothed_defence / avg


    strength_df = pd.DataFrame({
        'team': goals.keys(),
        'attack': goals.values(),
        'defence': conceded.values()
        })
    return strength_df


def predict_match(team1, team2, ratings_df, avg_goals):

    team_one_attack = ratings_df[ratings_df['team'] == team1]['attack'].values[0]
    team_one_defence = ratings_df[ratings_df['team'] == team1]['defence'].values[0]
    team_two_attack = ratings_df[ratings_df['team'] == team2]['attack'].values[0]
    team_two_defence = ratings_df[ratings_df['team'] == team2]['defence'].values[0]

    
    lambda1 = team_one_attack * team_two_defence * avg_goals
    lambda2 = team_two_attack * team_one_defence * avg_goals

    goals_range = np.arange(0, 8)
    team1_probs = poisson.pmf(goals_range, lambda1)
    team2_probs = poisson.pmf(goals_range, lambda2)
    matrix = np.outer(team1_probs, team2_probs)

    draw = np.trace(matrix)
    team1_win = np.tril(matrix, -1).sum()
    team2_win = np.triu(matrix, 1).sum()

    return {
        'team1': team1,
        'team2': team2,
        'prob_team1_win': team1_win,
        'prob_draw': draw,
        'prob_team2_win': team2_win,
        'lambda1': lambda1,
        'lambda2': lambda2
    }

def scoreline_matrix(lambda1, lambda2, max_goals = 5):
    k = np.arange(max_goals)
    probs_team1 = poisson.pmf(k, mu=lambda1)
    probs_team2 = poisson.pmf(k, mu=lambda2)
    scoreline_prob = np.outer(probs_team1, probs_team2)
    return scoreline_prob
