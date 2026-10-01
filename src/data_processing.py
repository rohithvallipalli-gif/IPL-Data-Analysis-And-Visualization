"""
data_processing.py
------------------
Data loading, inspection, cleaning, wrangling, outlier analysis,
normalization, and standardization for IPL dataset (2008-2024).
"""

import os
import pandas as pd
import numpy as np


TEAM_NAME_STANDARDIZATION = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiants': 'Rising Pune Supergiant',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
}

VENUE_CITY_MAPPING = {
    'Dubai International Cricket Stadium': 'Dubai',
    'Sharjah Cricket Stadium': 'Sharjah',
}


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the IPL dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """Inspect basic structure, data types, missing values, and duplicates."""
    inspection = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_count': int(df.duplicated().sum()),
        'seasons': sorted([int(s) for s in df['season'].dropna().unique()]),
        'innings_counts': df['innings'].value_counts().to_dict(),
    }
    return inspection


def clean_and_wrangle_data(df: pd.DataFrame, standardize_teams: bool = True) -> pd.DataFrame:
    """
    Performs genuine data preprocessing on the IPL dataset:
    - Parses date to datetime format
    - Converts season to integer
    - Imputes missing cities for known UAE venues
    - Handles winner missing values (matches with no result)
    - Standardizes renamed franchise names if requested
    - Computes middle overs runs and boundary contributions
    - Identifies regular 20-over match innings (innings 1 & 2) vs super-overs
    """
    df_clean = df.copy()

    # 1. Date conversion
    df_clean['date'] = pd.to_datetime(df_clean['date'], format='%Y-%m-%d', errors='coerce')

    # 2. Season datatype conversion
    df_clean['season'] = df_clean['season'].astype(int)

    # 3. Impute missing cities based on venue
    for venue, city in VENUE_CITY_MAPPING.items():
        mask = (df_clean['venue'] == venue) & (df_clean['city'].isna())
        df_clean.loc[mask, 'city'] = city

    # 4. Standardize franchise names for historical consistency
    if standardize_teams:
        df_clean['batting_team_std'] = df_clean['batting_team'].replace(TEAM_NAME_STANDARDIZATION)
        df_clean['bowling_team_std'] = df_clean['bowling_team'].replace(TEAM_NAME_STANDARDIZATION)
        df_clean['toss_winner_std'] = df_clean['toss_winner'].replace(TEAM_NAME_STANDARDIZATION)
        df_clean['winner_std'] = df_clean['winner'].replace(TEAM_NAME_STANDARDIZATION)
    else:
        df_clean['batting_team_std'] = df_clean['batting_team']
        df_clean['bowling_team_std'] = df_clean['bowling_team']
        df_clean['toss_winner_std'] = df_clean['toss_winner']
        df_clean['winner_std'] = df_clean['winner']

    # 5. Method column labeling
    df_clean['match_result_type'] = df_clean['method'].apply(
        lambda x: 'D/L Rain Affected' if str(x).strip().upper() == 'D/L' else 'Normal'
    )

    # 6. Flag regular innings (innings 1 and 2) vs super overs (innings 3, 4, 5, 6)
    df_clean['is_regular_innings'] = df_clean['innings'].isin([1, 2])

    # 7. Derived metrics
    # Middle overs runs: total_runs - powerplay_runs - death_over_runs
    df_clean['middle_over_runs'] = (
        df_clean['total_runs'] - df_clean['powerplay_runs'] - df_clean['death_over_runs']
    ).clip(lower=0)

    # Boundary metrics
    df_clean['boundary_runs'] = df_clean['boundaries_4'] * 4 + df_clean['boundaries_6'] * 6
    df_clean['boundary_run_pct'] = np.where(
        df_clean['total_runs'] > 0,
        (df_clean['boundary_runs'] / df_clean['total_runs']) * 100.0,
        0.0
    )

    return df_clean


def detect_outliers_iqr(series: pd.Series) -> dict:
    """Calculate IQR bounds and identify outlier boundaries."""
    q25 = float(np.percentile(series.dropna(), 25))
    q75 = float(np.percentile(series.dropna(), 75))
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return {
        'q25': round(q25, 2),
        'median': round(float(np.median(series.dropna())), 2),
        'q75': round(q75, 2),
        'iqr': round(iqr, 2),
        'lower_bound': round(lower_bound, 2),
        'upper_bound': round(upper_bound, 2),
        'outlier_count': int(len(outliers)),
        'outlier_percentage': round((len(outliers) / len(series)) * 100, 2)
    }


def normalize_min_max(series: pd.Series) -> pd.Series:
    """Normalize numeric feature to [0, 1] range using Min-Max scaling."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def standardize_z_score(series: pd.Series) -> pd.Series:
    """Standardize numeric feature to mean 0, variance 1 using Z-score scaling."""
    mean_val = series.mean()
    std_val = series.std(ddof=0)
    if std_val == 0:
        return pd.Series(0.0, index=series.index)
    return (series - mean_val) / std_val
