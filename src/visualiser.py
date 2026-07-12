import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_scoreline_heatmap(matrix, team1, team2):
    """
    Plots an 8x8 heatmap of scoreline probabilities for team1 vs team2.
    """
    plt.figure(figsize=(8, 6))  # width, height in inches

    labels = np.where(matrix < 0.005, '', np.round(matrix * 100, 1).astype(str) + '%')

    sns.heatmap(
        matrix,
        annot=labels,   # your custom string labels instead of the raw numbers
        fmt='',         # empty because labels are already strings, not numbers to format
        cmap='YlOrRd',
        cbar_kws={'label': 'Probability'}
    )

    plt.xlabel(f'{team2} goals')
    plt.ylabel(f'{team1} goals')
    plt.title(f'Scoreline Probability: {team1} vs {team2}')
    plt.tight_layout()  # prevents labels getting cut off at the edges
    plt.show()


def plot_predictions_summary(predictions_df):
    n = len(predictions_df)
    y = np.arange(n)
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(8, 0.6 * n + 2))

    bars1 = ax.barh(y + bar_height/2, predictions_df['prob_team1_win'],
                     height=bar_height, color='#1f77b4')
    bars2 = ax.barh(y - bar_height/2, predictions_df['prob_team2_win'],
                     height=bar_height, color='#ff7f0e')

    # label each bar with the team name + probability
    for bar, team, prob in zip(bars1, predictions_df['team1'], predictions_df['prob_team1_win']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{team} ({prob:.0%})', va='center', fontsize=9)

    for bar, team, prob in zip(bars2, predictions_df['team2'], predictions_df['prob_team2_win']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{team} ({prob:.0%})', va='center', fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels([f'Match {i+1}' for i in range(n)])  # or blank, since bars are self-labeled

    ax.set_xlabel('Win Probability')
    ax.set_title('Match Predictions: Win Probability')
    ax.set_xlim(0, 1.15)  # extra room so text labels don't get cut off
    plt.tight_layout()
    plt.show()