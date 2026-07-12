import requests
import pandas as pd

def load_matches(url):
    try:
        response = requests.get(url)
        data = response.json()
        completed_matches = [m for m in data['matches'] if m.get('score') is not None]
        df = pd.DataFrame(completed_matches)

        df['ft'] = df['score'].apply(lambda s: s['ft'])
        df['goal1'] = df['ft'].apply(lambda s: s[0])
        df['goal2'] = df['ft'].apply(lambda s: s[1])
        df = df[['round', 'date', 'team1', 'team2', 'group', 'goal1', 'goal2']]
    except Exception as e:
        print(f"Error: {e}")
        return None

    return df