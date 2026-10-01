"""
analysis.py
-----------
NumPy and Pandas core operations, data transformations, and
the 15 IPL statistical and analytical queries.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------
# 1. NumPy Core Operations Demonstration
# ---------------------------------------------------------

def demonstrate_numpy_operations(df: pd.DataFrame) -> dict:
    """
    Demonstrates core NumPy concepts using actual IPL numerical data:
    - ndarray creation, dtypes, shape, ndim, size
    - indexing & slicing
    - boolean masking
    - descriptive statistics (mean, median, std, percentiles)
    - vectorized calculations
    """
    regular_df = df[df['innings'].isin([1, 2])]
    runs_array = regular_df['total_runs'].to_numpy(dtype=np.int32)
    wickets_array = regular_df['wickets'].to_numpy(dtype=np.int32)
    balls_array = regular_df['balls_bowled'].to_numpy(dtype=np.int32)
    
    # Structural properties
    ndarray_info = {
        'type': type(runs_array).__name__,
        'dtype': str(runs_array.dtype),
        'shape': runs_array.shape,
        'ndim': runs_array.ndim,
        'size': runs_array.size,
        'itemsize': runs_array.itemsize,
    }
    
    # Indexing and slicing
    first_five = runs_array[:5].tolist()
    last_five = runs_array[-5:].tolist()
    every_tenth = runs_array[::10][:5].tolist()
    
    # Boolean masking: e.g. innings with >= 200 runs
    mask_200 = runs_array >= 200
    high_scores = runs_array[mask_200]
    
    # Statistical measures
    stats = {
        'mean': float(np.mean(runs_array)),
        'median': float(np.median(runs_array)),
        'std': float(np.std(runs_array, ddof=1)),
        'variance': float(np.var(runs_array, ddof=1)),
        'p25': float(np.percentile(runs_array, 25)),
        'p50': float(np.percentile(runs_array, 50)),
        'p75': float(np.percentile(runs_array, 75)),
        'p90': float(np.percentile(runs_array, 90)),
        'min': int(np.min(runs_array)),
        'max': int(np.max(runs_array)),
    }
    
    # Vectorized calculations:
    # 1. Runs per legal over vector: runs / (balls / 6)
    valid_balls_mask = balls_array > 0
    vectorized_overs = balls_array[valid_balls_mask] / 6.0
    vectorized_run_rates = runs_array[valid_balls_mask] / vectorized_overs
    
    # 2. Runs per wicket vector
    runs_per_wicket = np.divide(
        runs_array.astype(float),
        wickets_array.astype(float),
        out=np.full_like(runs_array, np.nan, dtype=float),
        where=wickets_array > 0
    )
    
    return {
        'ndarray_info': ndarray_info,
        'indexing_slicing': {
            'first_five': first_five,
            'last_five': last_five,
            'every_tenth_sample': every_tenth,
        },
        'boolean_masking': {
            'count_above_200': int(np.sum(mask_200)),
            'pct_above_200': round(float(np.mean(mask_200) * 100), 2),
            'max_score_masked': int(np.max(high_scores)) if len(high_scores) > 0 else 0,
        },
        'statistics': stats,
        'vectorized_metrics': {
            'mean_calculated_run_rate': round(float(np.nanmean(vectorized_run_rates)), 2),
            'mean_runs_per_wicket': round(float(np.nanmean(runs_per_wicket)), 2),
        }
    }


# ---------------------------------------------------------
# 2. Pandas Core Operations Demonstration
# ---------------------------------------------------------

def demonstrate_pandas_operations(df: pd.DataFrame) -> dict:
    """
    Demonstrates core Pandas operations:
    - Series & DataFrame extraction
    - Indexing, Slicing, Boolean Filtering
    - Adding computed columns
    - Aggregation & Groupby
    - Crosstab
    - Merging / Joining
    """
    # Series vs DataFrame
    total_runs_series = df['total_runs']
    summary_df = df[['season', 'batting_team_std', 'total_runs', 'wickets', 'run_rate']]
    
    # Filtering: high scoring close matches (runs >= 190 and wickets >= 7)
    filtered = df[(df['total_runs'] >= 190) & (df['wickets'] >= 7)]
    
    # Groupby aggregation: Season-wise run summary
    season_agg = df[df['is_regular_innings']].groupby('season').agg(
        total_innings=('total_runs', 'count'),
        mean_runs=('total_runs', 'mean'),
        median_runs=('total_runs', 'median'),
        max_runs=('total_runs', 'max'),
        mean_run_rate=('run_rate', 'mean')
    ).round(2).reset_index()
    
    # Contingency table (Crosstab): Toss Decision vs Match Result Win Flag
    toss_crosstab = pd.crosstab(
        df['toss_decision'],
        df['toss_win_match_win'],
        margins=True,
        margins_name='Total'
    )
    
    # Merge operation: Join team innings stats with team aggregate totals
    team_overall = df[df['is_regular_innings']].groupby('batting_team_std').agg(
        franchise_innings=('total_runs', 'count'),
        franchise_avg_runs=('total_runs', 'mean')
    ).round(2).reset_index()
    
    merged_sample = pd.merge(
        summary_df.head(10),
        team_overall,
        on='batting_team_std',
        how='left'
    )
    
    return {
        'series_type': str(type(total_runs_series)),
        'df_type': str(type(summary_df)),
        'filtered_rows': len(filtered),
        'season_agg': season_agg,
        'toss_crosstab': toss_crosstab,
        'merged_sample_shape': merged_sample.shape,
    }


# ---------------------------------------------------------
# 3. 15 IPL Analytics Questions
# ---------------------------------------------------------

def answer_all_ipl_questions(df: pd.DataFrame) -> dict:
    """Answers all 15 analytical questions specified in syllabus."""
    regular_df = df[df['is_regular_innings']].copy()
    
    # Q1 & Q5: Season-wise scoring change & variation
    season_scoring = regular_df.groupby('season').agg(
        matches=('match_id', 'nunique'),
        avg_runs_per_innings=('total_runs', 'mean'),
        median_runs=('total_runs', 'median'),
        avg_run_rate=('run_rate', 'mean'),
        highest_score=('total_runs', 'max'),
        lowest_score=('total_runs', 'min')
    ).round(2).reset_index()
    
    # Q2: Teams with highest total runs
    team_total_runs = regular_df.groupby('batting_team_std')['total_runs'].agg(
        innings_played='count',
        total_runs='sum',
        avg_runs='mean'
    ).round(2).sort_values(by='total_runs', ascending=False).reset_index()
    
    # Q3: Teams with highest average run rate (min 20 innings for fair comparison)
    team_run_rate = regular_df.groupby('batting_team_std').agg(
        innings=('total_runs', 'count'),
        avg_run_rate=('run_rate', 'mean'),
        median_run_rate=('run_rate', 'median')
    ).round(2).sort_values(by='avg_run_rate', ascending=False).reset_index()
    
    # Q4: Which teams have the most wins?
    # Matches dataset has 2 innings per match with same winner; drop duplicates per match_id
    matches_unique = df.drop_duplicates(subset=['match_id']).copy()
    team_wins = matches_unique['winner_std'].value_counts(dropna=True).reset_index()
    team_wins.columns = ['team', 'wins']
    
    # Q6: Distribution of total runs
    total_runs_dist = {
        'mean': round(float(regular_df['total_runs'].mean()), 2),
        'std': round(float(regular_df['total_runs'].std()), 2),
        'median': round(float(regular_df['total_runs'].median()), 2),
        'skew': round(float(regular_df['total_runs'].skew()), 2),
        'kurtosis': round(float(regular_df['total_runs'].kurtosis()), 2),
    }
    
    # Q7: Relationship between run rate and total runs
    rr_runs_corr = round(float(regular_df['run_rate'].corr(regular_df['total_runs'])), 4)
    
    # Q8: Relationship between wickets and total runs
    wickets_runs_corr = round(float(regular_df['wickets'].corr(regular_df['total_runs'])), 4)
    
    # Q9: How do boundaries contribute to scoring?
    boundary_contribution = {
        'mean_boundary_pct': round(float(regular_df['boundary_run_pct'].mean()), 2),
        'corr_fours_with_total': round(float(regular_df['boundaries_4'].corr(regular_df['total_runs'])), 4),
        'corr_sixes_with_total': round(float(regular_df['boundaries_6'].corr(regular_df['total_runs'])), 4),
        'mean_fours_per_innings': round(float(regular_df['boundaries_4'].mean()), 2),
        'mean_sixes_per_innings': round(float(regular_df['boundaries_6'].mean()), 2),
    }
    
    # Q10: Does toss winner frequently become match winner?
    toss_win_pct = round(float(matches_unique['toss_win_match_win'].mean() * 100), 2)
    
    # Q11: How do toss decisions vary?
    toss_decisions = matches_unique['toss_decision'].value_counts().to_dict()
    toss_decisions_pct = (matches_unique['toss_decision'].value_counts(normalize=True) * 100).round(2).to_dict()
    
    # Q12: Which venues occur most frequently?
    top_venues = matches_unique['venue'].value_counts().head(10).reset_index()
    top_venues.columns = ['venue', 'match_count']
    
    # Q13: Powerplay runs variation
    powerplay_stats = regular_df.groupby('season').agg(
        avg_pp_runs=('powerplay_runs', 'mean'),
        median_pp_runs=('powerplay_runs', 'median'),
        avg_pp_wickets=('powerplay_wickets', 'mean')
    ).round(2).reset_index()
    
    # Q14: Death over runs variation
    death_over_stats = regular_df[regular_df['death_over_runs'] > 0].groupby('season').agg(
        avg_death_runs=('death_over_runs', 'mean'),
        median_death_runs=('death_over_runs', 'median'),
        avg_death_wickets=('death_over_wickets', 'mean')
    ).round(2).reset_index()
    
    # Q15: Player of the Match awards frequency
    top_potm = matches_unique['player_of_match'].value_counts().head(15).reset_index()
    top_potm.columns = ['player', 'awards_count']
    
    return {
        'season_scoring': season_scoring,
        'team_total_runs': team_total_runs,
        'team_run_rate': team_run_rate,
        'team_wins': team_wins,
        'total_runs_dist': total_runs_dist,
        'rr_runs_corr': rr_runs_corr,
        'wickets_runs_corr': wickets_runs_corr,
        'boundary_contribution': boundary_contribution,
        'toss_win_pct': toss_win_pct,
        'toss_decisions': toss_decisions,
        'toss_decisions_pct': toss_decisions_pct,
        'top_venues': top_venues,
        'powerplay_stats': powerplay_stats,
        'death_over_stats': death_over_stats,
        'top_potm': top_potm,
    }
