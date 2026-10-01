"""
visualization.py
----------------
Generates all 20 required academic visualizations using Matplotlib and Seaborn.
Saves high-resolution figures to outputs/figures/.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import squarify

# Style settings for clean academic presentation
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#E0E0E0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


# -------------------------------------------------------------
# Basic Visualizations (1-12)
# -------------------------------------------------------------

def plot_season_scoring_trend(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """1. Season-wise scoring trend — Line Plot"""
    regular_df = df[df['is_regular_innings']]
    season_data = regular_df.groupby('season')['total_runs'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(season_data['season'], season_data['total_runs'], marker='o', color='#1f77b4', linewidth=2, markersize=6)
    ax.set_title('Season-Wise Average Innings Score (2008–2024)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('IPL Season', fontsize=11)
    ax.set_ylabel('Average Total Runs per Innings', fontsize=11)
    ax.set_xticks(season_data['season'])
    ax.set_xticklabels(season_data['season'], rotation=45)
    ax.grid(True)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, '01_season_scoring_trend.png'), dpi=300)
    return fig


def plot_team_total_runs(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """2. Team-wise total runs — Bar Plot"""
    regular_df = df[df['is_regular_innings']]
    team_runs = regular_df.groupby('batting_team_std')['total_runs'].sum().sort_values(ascending=False).reset_index()
    
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    sns.barplot(data=team_runs, x='total_runs', y='batting_team_std', hue='batting_team_std', palette='Blues_r', legend=False, ax=ax)
    ax.set_title('Aggregate Total Runs by Franchise (2008–2024)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Total Runs Scored', fontsize=11)
    ax.set_ylabel('Franchise', fontsize=11)
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{int(width):,}", (width + 300, p.get_y() + p.get_height() / 2),
                    ha='left', va='center', fontsize=9, color='#333333')
    ax.grid(axis='x')
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, '02_team_total_runs.png'), dpi=300)
    return fig


def plot_team_avg_run_rate(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """3. Team-wise average run rate — Bar Plot"""
    regular_df = df[df['is_regular_innings']]
    team_rr = regular_df.groupby('batting_team_std')['run_rate'].mean().sort_values(ascending=False).reset_index()
    
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    sns.barplot(data=team_rr, x='run_rate', y='batting_team_std', hue='batting_team_std', palette='mako', legend=False, ax=ax)
    ax.set_title('Historical Average Run Rate by Franchise', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Average Run Rate (Runs per Over)', fontsize=11)
    ax.set_ylabel('Franchise', fontsize=11)
    ax.set_xlim(6.0, 9.5)
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{width:.2f}", (width + 0.05, p.get_y() + p.get_height() / 2),
                    ha='left', va='center', fontsize=9)
    ax.grid(axis='x')
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, '03_team_avg_run_rate.png'), dpi=300)
    return fig


def plot_total_runs_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """4. Total runs distribution — Histogram"""
    regular_df = df[df['is_regular_innings']]
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    sns.histplot(regular_df['total_runs'], kde=True, bins=30, color='#2b5c8f', ax=ax)
    median_val = regular_df['total_runs'].median()
    mean_val = regular_df['total_runs'].mean()
    ax.axvline(median_val, color='#d95f02', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
    ax.axvline(mean_val, color='#7570b3', linestyle=':', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.set_title('Distribution of Innings Total Runs (Regular Innings)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Total Runs', fontsize=11)
    ax.set_ylabel('Frequency (Innings Count)', fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, '04_total_runs_distribution.png'), dpi=300)
    return fig


def plot_run_rate_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """5. Run-rate distribution — Histogram"""
    regular_df = df[df['is_regular_innings']]
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    sns.histplot(regular_df['run_rate'], kde=True, bins=35, color='#1b9e77', ax=ax)
    ax.set_title('Distribution of Innings Run Rates', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Run Rate (Runs per Over)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.grid(True)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, '05_run_rate_distribution.png'), dpi=300)
    return fig


def plot_run_rate_vs_total_runs(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """6. Run Rate vs Total Runs — Scatter Plot"""
    regular_df = df[df['is_regular_innings']]
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sns.scatterplot(data=regular_df, x='run_rate', y='total_runs', alpha=0.5, color='#386cb0', ax=ax)
    sns.regplot(data=regular_df, x='run_rate', y='total_runs', scatter=False, color='#e41a1c', ax=ax, line_kws={'linewidth': 2})
    ax.set_title('Correlation Between Run Rate and Total Runs', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Run Rate (Runs per Over)', fontsize=11)
    ax.set_ylabel('Total Runs Scored', fontsize=11)
    ax.grid(True)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, '06_run_rate_vs_total_runs.png'), dpi=300)
    return fig


def plot_wickets_vs_total_runs(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """7. Wickets vs Total Runs — Scatter Plot with Trend"""
    regular_df = df[df['is_regular_innings']]
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sns.boxplot(data=regular_df, x='wickets', y='total_runs', color='#a6bddb', ax=ax)
    ax.set_title('Innings Total Runs Across Wickets Lost', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Wickets Lost', fontsize=11)
    ax.set_ylabel('Total Runs', fontsize=11)
    ax.grid(axis='y')
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, '07_wickets_vs_total_runs.png'), dpi=300)
    return fig


def plot_dot_ball_pct_vs_run_rate(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """8. Dot Ball Percentage vs Run Rate — Scatter Plot"""
    regular_df = df[df['is_regular_innings']]
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sns.scatterplot(data=regular_df, x='dot_ball_pct', y='run_rate', alpha=0.5, color='#4daf4a', ax=ax)
    sns.regplot(data=regular_df, x='dot_ball_pct', y='run_rate', scatter=False, color='#e41a1c', ax=ax, line_kws={'linewidth': 2})
    ax.set_title('Impact of Dot Ball Percentage on Run Rate', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Dot Ball Percentage (%)', fontsize=11)
    ax.set_ylabel('Innings Run Rate', fontsize=11)
    ax.grid(True)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, '08_dot_ball_pct_vs_run_rate.png'), dpi=300)
    return fig


def plot_player_of_match_frequency(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """9. Player of the Match frequency — Bar Plot"""
    matches_unique = df.drop_duplicates(subset=['match_id'])
    top_players = matches_unique['player_of_match'].value_counts().head(12).reset_index()
    top_players.columns = ['player', 'awards']
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    sns.barplot(data=top_players, x='awards', y='player', hue='player', palette='viridis', legend=False, ax=ax)
    ax.set_title('Top 12 Most Frequent Player of the Match Recipients', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Awards Count', fontsize=11)
    ax.set_ylabel('Player', fontsize=11)
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{int(width)}", (width + 0.3, p.get_y() + p.get_height() / 2),
                    ha='left', va='center', fontsize=9)
    ax.grid(axis='x')
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, '09_player_of_match_frequency.png'), dpi=300)
    return fig


def plot_venue_frequency(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """10. Venue frequency — Bar Plot"""
    matches_unique = df.drop_duplicates(subset=['match_id'])
    top_venues = matches_unique['venue'].value_counts().head(10).reset_index()
    top_venues.columns = ['venue', 'matches']
    
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    sns.barplot(data=top_venues, x='matches', y='venue', hue='venue', palette='rocket', legend=False, ax=ax)
    ax.set_title('Top 10 Most Frequently Hosted IPL Stadiums', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Matches Hosted', fontsize=11)
    ax.set_ylabel('Stadium Venue', fontsize=11)
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{int(width)}", (width + 1.0, p.get_y() + p.get_height() / 2),
                    ha='left', va='center', fontsize=9)
    ax.grid(axis='x')
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, '10_venue_frequency.png'), dpi=300)
    return fig


def plot_toss_decision_pie(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """11. Toss decision distribution — Pie Chart"""
    matches_unique = df.drop_duplicates(subset=['match_id'])
    counts = matches_unique['toss_decision'].value_counts()
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    colors = ['#4a7bb0', '#e08244']
    ax.pie(counts.values, labels=counts.index.str.capitalize(), autopct='%1.1f%%',
           startangle=140, colors=colors, textprops={'fontsize': 11})
    ax.set_title('Overall Toss Decision Distribution', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, '11_toss_decision_distribution.png'), dpi=300)
    return fig


def plot_related_metrics_subplots(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """12. Related IPL metrics using Subplots (4-panel overview)"""
    regular_df = df[df['is_regular_innings']]
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)
    
    # Subplot 1: Powerplay Runs vs Total Runs
    sns.scatterplot(data=regular_df, x='powerplay_runs', y='total_runs', alpha=0.4, color='#377eb8', ax=axes[0, 0])
    axes[0, 0].set_title('Powerplay Runs vs Total Runs', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Powerplay Runs')
    axes[0, 0].set_ylabel('Total Runs')
    axes[0, 0].grid(True)
    
    # Subplot 2: Death Over Runs vs Total Runs
    death_df = regular_df[regular_df['death_over_runs'] > 0]
    sns.scatterplot(data=death_df, x='death_over_runs', y='total_runs', alpha=0.4, color='#e41a1c', ax=axes[0, 1])
    axes[0, 1].set_title('Death Over Runs vs Total Runs', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Death Over Runs')
    axes[0, 1].set_ylabel('Total Runs')
    axes[0, 1].grid(True)
    
    # Subplot 3: Boundaries (4s and 6s) Count vs Total Runs
    sns.scatterplot(data=regular_df, x='boundaries_6', y='total_runs', alpha=0.4, color='#984ea3', ax=axes[1, 0])
    axes[1, 0].set_title('Sixes Count vs Total Runs', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Sixes Struck')
    axes[1, 0].set_ylabel('Total Runs')
    axes[1, 0].grid(True)
    
    # Subplot 4: Innings 1 vs Innings 2 Runs Distribution
    sns.kdeplot(data=regular_df, x='total_runs', hue='innings', common_norm=False, palette='Set1', ax=axes[1, 1])
    axes[1, 1].set_title('Total Runs: Innings 1 vs Innings 2', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Total Runs')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, '12_related_metrics_subplots.png'), dpi=300)
    return fig


# -------------------------------------------------------------
# Statistical Visualizations (13-14)
# -------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """13. Correlation Heatmap"""
    regular_df = df[df['is_regular_innings']]
    num_cols = [
        'total_runs', 'wickets', 'boundaries_4', 'boundaries_6',
        'dot_balls', 'run_rate', 'powerplay_runs', 'powerplay_wickets',
        'death_over_runs', 'death_over_wickets', 'extras'
    ]
    corr_matrix = regular_df[num_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={'label': 'Pearson Correlation'}, ax=ax)
    ax.set_title('Correlation Matrix of Core IPL Innings Numerical Features', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, '13_correlation_heatmap.png'), dpi=300)
    return fig


def plot_numerical_boxplots(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """14. Box Plot for important numerical variables (Outlier Detection)"""
    regular_df = df[df['is_regular_innings']]
    vars_to_plot = ['total_runs', 'powerplay_runs', 'death_over_runs', 'dot_balls']
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 5), dpi=150)
    for i, var in enumerate(vars_to_plot):
        sns.boxplot(y=regular_df[var], ax=axes[i], color='#7294d4')
        axes[i].set_title(var.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Value')
        axes[i].grid(axis='y')
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, '14_numerical_boxplots.png'), dpi=300)
    return fig


# -------------------------------------------------------------
# Advanced Visualizations (15-20)
# -------------------------------------------------------------

def plot_crosstab_visualization(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """15. Crosstab visualization — Heatmap of Season vs Toss Decision"""
    matches_unique = df.drop_duplicates(subset=['match_id'])
    ct = pd.crosstab(matches_unique['season'], matches_unique['toss_decision'], normalize='index') * 100
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    sns.heatmap(ct, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': 'Percentage (%)'}, ax=ax)
    ax.set_title('Crosstab Visualization: Toss Decision Proportion by Season', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Toss Decision', fontsize=11)
    ax.set_ylabel('IPL Season', fontsize=11)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, '15_crosstab_visualization.png'), dpi=300)
    return fig


def plot_treemap_team_runs(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """16. Treemap — Team share of total runs scored across IPL history"""
    regular_df = df[df['is_regular_innings']]
    team_runs = regular_df.groupby('batting_team_std')['total_runs'].sum().reset_index()
    team_runs = team_runs.sort_values(by='total_runs', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    cmap = plt.cm.tab20
    colors = [cmap(i / len(team_runs)) for i in range(len(team_runs))]
    labels = [f"{row['batting_team_std']}\n{int(row['total_runs']):,} runs" for _, row in team_runs.iterrows()]
    
    squarify.plot(sizes=team_runs['total_runs'], label=labels, color=colors, alpha=0.85, ax=ax,
                  text_kwargs={'fontsize': 8, 'fontweight': 'bold'})
    ax.set_title('Treemap: Franchise Proportions of Cumulative IPL Runs', fontsize=13, fontweight='bold', pad=12)
    ax.axis('off')
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, '16_treemap_team_runs.png'), dpi=300)
    return fig


def plot_bump_ranking_chart(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """17. Season-wise team ranking / bump-style chart (Top 6 active franchises wins per season)"""
    matches_unique = df.drop_duplicates(subset=['match_id']).copy()
    top_teams = ['Mumbai Indians', 'Chennai Super Kings', 'Kolkata Knight Riders',
                 'Royal Challengers Bengaluru', 'Delhi Capitals', 'Punjab Kings']
    
    matches_top = matches_unique[matches_unique['winner_std'].isin(top_teams)]
    season_wins = matches_top.groupby(['season', 'winner_std']).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    palette = sns.color_palette('tab10', n_colors=len(top_teams))
    for i, team in enumerate(top_teams):
        if team in season_wins.columns:
            ax.plot(season_wins.index, season_wins[team], marker='o', linewidth=2.2, label=team, color=palette[i])
    
    ax.set_title('Season-Wise Annual Match Wins by Key Franchises', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Season', fontsize=11)
    ax.set_ylabel('Matches Won in Season', fontsize=11)
    ax.set_xticks(season_wins.index)
    ax.set_xticklabels(season_wins.index, rotation=45)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    ax.grid(True)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, '17_bump_ranking_chart.png'), dpi=300)
    return fig


def plot_waterfall_innings_breakdown(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """18. Waterfall chart — Innings run buildup decomposition"""
    regular_df = df[df['is_regular_innings']]
    avg_pp = regular_df['powerplay_runs'].mean()
    avg_middle = regular_df['middle_over_runs'].mean()
    avg_death = regular_df['death_over_runs'].mean()
    avg_extras = regular_df['extras'].mean()
    avg_total = regular_df['total_runs'].mean()
    
    categories = ['Powerplay\n(Overs 1-6)', 'Middle Overs\n(Overs 7-15)', 'Death Overs\n(Overs 16-20)', 'Extras', 'Average\nTotal Runs']
    amounts = [avg_pp, avg_middle, avg_death, avg_extras, avg_total]
    
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    
    # Calculate bottom baseline for waterfall steps
    bottoms = [0, avg_pp, avg_pp + avg_middle, avg_pp + avg_middle + avg_death, 0]
    heights = [avg_pp, avg_middle, avg_death, avg_extras, avg_total]
    colors = ['#4575b4', '#74add1', '#f46d43', '#fdae61', '#313695']
    
    bars = ax.bar(categories, heights, bottom=bottoms, color=colors, width=0.55, edgecolor='#333333')
    
    for i, bar in enumerate(bars):
        val = amounts[i]
        top = bottoms[i] + heights[i]
        ax.annotate(f"{val:.1f}", (bar.get_x() + bar.get_width() / 2, top + 2),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_title('Waterfall Chart: Average Innings Run Accumulation Phase Breakdown', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('Runs Contributed', fontsize=11)
    ax.set_ylim(0, 180)
    ax.grid(axis='y')
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, '18_waterfall_innings_breakdown.png'), dpi=300)
    return fig


def plot_waffle_chart_toss(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """19. Waffle chart — 10x10 matrix of Toss Decision proportions"""
    matches_unique = df.drop_duplicates(subset=['match_id'])
    field_pct = int(round(matches_unique['toss_decision'].value_counts(normalize=True).get('field', 0.6) * 100))
    bat_pct = 100 - field_pct
    
    # 10x10 grid representation
    grid = np.zeros((10, 10))
    for i in range(100):
        r, c = divmod(i, 10)
        grid[r, c] = 1 if i < field_pct else 2
        
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    cmap = plt.matplotlib.colors.ListedColormap(['#2b83ba', '#fdae61'])
    ax.pcolor(grid, cmap=cmap, edgecolors='white', linewidths=2)
    ax.set_title(f'Waffle Chart: Toss Decision Proportions\nField ({field_pct}%) vs Bat ({bat_pct}%)',
                 fontsize=12, fontweight='bold', pad=12)
    ax.axis('off')
    
    # Legend patches
    patch_field = plt.matplotlib.patches.Patch(color='#2b83ba', label=f'Field First ({field_pct}%)')
    patch_bat = plt.matplotlib.patches.Patch(color='#fdae61', label=f'Bat First ({bat_pct}%)')
    ax.legend(handles=[patch_field, patch_bat], loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=True)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, '19_waffle_chart_toss.png'), dpi=300)
    return fig


def plot_geographic_city_matches(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """20. Geographic visualization — Matches hosted across cities (India and UAE)"""
    matches_unique = df.drop_duplicates(subset=['match_id'])
    city_counts = matches_unique['city'].dropna().value_counts().head(12).reset_index()
    city_counts.columns = ['city', 'matches']
    
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    sns.barplot(data=city_counts, x='matches', y='city', hue='city', palette='crest', legend=False, ax=ax)
    ax.set_title('Geographic Distribution: Matches Hosted by Host City (India & UAE)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Number of Matches', fontsize=11)
    ax.set_ylabel('City', fontsize=11)
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{int(width)}", (width + 1.5, p.get_y() + p.get_height() / 2),
                    ha='left', va='center', fontsize=9)
    ax.grid(axis='x')
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, '20_geographic_city_matches.png'), dpi=300)
    return fig


def generate_all_visualizations(df: pd.DataFrame) -> dict:
    """Generates and saves all 20 visualizations."""
    figs = {
        '01_season_scoring_trend': plot_season_scoring_trend(df),
        '02_team_total_runs': plot_team_total_runs(df),
        '03_team_avg_run_rate': plot_team_avg_run_rate(df),
        '04_total_runs_distribution': plot_total_runs_distribution(df),
        '05_run_rate_distribution': plot_run_rate_distribution(df),
        '06_run_rate_vs_total_runs': plot_run_rate_vs_total_runs(df),
        '07_wickets_vs_total_runs': plot_wickets_vs_total_runs(df),
        '08_dot_ball_pct_vs_run_rate': plot_dot_ball_pct_vs_run_rate(df),
        '09_player_of_match_frequency': plot_player_of_match_frequency(df),
        '10_venue_frequency': plot_venue_frequency(df),
        '11_toss_decision_distribution': plot_toss_decision_pie(df),
        '12_related_metrics_subplots': plot_related_metrics_subplots(df),
        '13_correlation_heatmap': plot_correlation_heatmap(df),
        '14_numerical_boxplots': plot_numerical_boxplots(df),
        '15_crosstab_visualization': plot_crosstab_visualization(df),
        '16_treemap_team_runs': plot_treemap_team_runs(df),
        '17_bump_ranking_chart': plot_bump_ranking_chart(df),
        '18_waterfall_innings_breakdown': plot_waterfall_innings_breakdown(df),
        '19_waffle_chart_toss': plot_waffle_chart_toss(df),
        '20_geographic_city_matches': plot_geographic_city_matches(df),
    }
    return figs
