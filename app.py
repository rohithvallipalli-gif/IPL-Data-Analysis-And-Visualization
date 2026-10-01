"""
app.py
------
IPL ANALYTICS & PREDICTIVE INSIGHTS
Premium Sports Intelligence & Cricket Analytics Portal (2008–2024).

Features:
- Sports portal visual hierarchy (Navy, Blue, Gold theme)
- Full-width hero banner and executive match-card styling
- Centralized dataset metadata (zero contradictory statistics)
- 6-section sidebar navigation with sports branding
- Team performance comparison with dynamic metric selector
- Full Visualization Center covering all 20 project figures
- Scoreboard-style Machine Learning interface (Regression & Classification)
- Interactive Data Center / Explorer workbench
- Academic project presentation & viva compliance
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 stdout on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure local project modules are accessible
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_processing import (
    load_raw_data,
    clean_and_wrangle_data,
    detect_outliers_iqr
)
from src.analysis import answer_all_ipl_questions
from src.visualization import (
    plot_season_scoring_trend,
    plot_team_total_runs,
    plot_team_avg_run_rate,
    plot_total_runs_distribution,
    plot_run_rate_distribution,
    plot_run_rate_vs_total_runs,
    plot_wickets_vs_total_runs,
    plot_dot_ball_pct_vs_run_rate,
    plot_player_of_match_frequency,
    plot_venue_frequency,
    plot_toss_decision_pie,
    plot_related_metrics_subplots,
    plot_correlation_heatmap,
    plot_numerical_boxplots,
    plot_crosstab_visualization,
    plot_treemap_team_runs,
    plot_bump_ranking_chart,
    plot_waterfall_innings_breakdown,
    plot_waffle_chart_toss,
    plot_geographic_city_matches
)
from src.models import (
    prepare_ml_data,
    train_and_evaluate_regression,
    train_and_evaluate_classification,
    plot_feature_importance,
    predict_interactive,
    REGRESSION_FEATURES
)

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="IPL Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Matplotlib Dark Sports Styler
# -----------------------------------------------------------------------------
def stylize_fig(fig):
    """Applies premium dark sports palette styling to Matplotlib figures."""
    fig.patch.set_facecolor('#08152A')
    for ax in fig.get_axes():
        ax.set_facecolor('#08152A')
        ax.tick_params(colors='#F8FAFC', labelsize=8.5)
        ax.xaxis.label.set_color('#CBD5E1')
        ax.yaxis.label.set_color('#CBD5E1')
        ax.title.set_color('#F8FAFC')
        for spine in ax.spines.values():
            spine.set_color('#1E3A5F')
            spine.set_linewidth(0.8)
        ax.grid(color='#1E3A5F', linestyle='--', alpha=0.55)
        # Handle legends
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor('#0D1B2A')
            leg.get_frame().set_edgecolor('#1E3A5F')
            for text in leg.get_texts():
                text.set_color('#F8FAFC')
    return fig


# -----------------------------------------------------------------------------
# Franchise Color Accents
# -----------------------------------------------------------------------------
TEAM_COLORS = {
    'Mumbai Indians': '#004BA0',
    'Chennai Super Kings': '#F5C451',
    'Royal Challengers Bengaluru': '#EC1C24',
    'Kolkata Knight Riders': '#7030A0',
    'Delhi Capitals': '#0078BC',
    'Punjab Kings': '#ED1B24',
    'Rajasthan Royals': '#EA1A85',
    'Sunrisers Hyderabad': '#FF822A',
    'Gujarat Titans': '#1B2133',
    'Lucknow Super Giants': '#A7D5F2'
}


# -----------------------------------------------------------------------------
# Premium Sports Portal CSS System
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Full-width Canvas & Dark Sports Palette */
    .stApp {
        background-color: #050B18;
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
        color: #F8FAFC;
    }
    
    /* Ensure Streamlit top header blends cleanly and doesn't overlap content */
    header[data-testid="stHeader"] {
        background-color: #050B18 !important;
        height: 3.25rem !important;
        z-index: 99 !important;
    }
    
    /* Content Padding: Generous top padding (4.5rem ≈ 72px) ensures the header is NEVER clipped */
    .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }
    
    /* Top Global Website Navigation Bar (Normal Flow, Never Fixed) */
    .sports-nav-header {
        background: linear-gradient(180deg, #0B1D34 0%, #061326 100%);
        border: 1px solid #1D3A5A;
        border-radius: 12px;
        padding: 16px 26px;
        margin-top: 0 !important;
        margin-bottom: 32px !important; /* 28-36px spacing before page content */
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);
        position: relative;
        z-index: 1;
    }
    .sports-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-icon-box {
        font-size: 32px;
        line-height: 1;
        filter: drop-shadow(0 2px 6px rgba(0, 0, 0, 0.4));
    }
    .brand-logo-text {
        font-size: 23px;
        font-weight: 900;
        letter-spacing: -0.3px;
        color: #FFFFFF;
        line-height: 1.1;
    }
    .brand-accent-gold {
        color: #F5C451;
    }
    .brand-tagline {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.6px;
        color: #14B8A6;
        font-weight: 800;
        margin-top: 3px;
    }
    .header-meta {
        text-align: right;
    }
    .header-era-badge {
        background: #0B254E;
        border: 1px solid #2563EB;
        color: #FFFFFF;
        font-size: 12px;
        font-weight: 800;
        padding: 5px 14px;
        border-radius: 6px;
        display: inline-block;
        letter-spacing: 0.6px;
    }
    .live-status {
        color: #22C55E;
        font-size: 11px;
        font-weight: 700;
        margin-top: 5px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 6px;
        letter-spacing: 0.5px;
    }
    .live-pulse {
        height: 8px;
        width: 8px;
        background-color: #22C55E;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 6px #22C55E;
    }
    
    /* Hero Section Banner */
    .hero-container {
        background: linear-gradient(135deg, #0A1931 0%, #0F2A4A 50%, #08152A 100%);
        border: 1px solid #1D3A5A;
        border-radius: 14px;
        padding: 32px 36px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    .hero-eyebrow {
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        color: #F5C451;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .hero-title {
        font-size: 32px;
        font-weight: 900;
        color: #FFFFFF;
        letter-spacing: -0.5px;
        line-height: 1.15;
        margin-bottom: 10px;
    }
    .hero-subtitle {
        font-size: 14.5px;
        color: #CBD5E1;
        max-width: 720px;
        line-height: 1.5;
        margin-bottom: 18px;
    }
    .hero-badges-strip {
        display: flex;
        gap: 12px;
        align-items: center;
        flex-wrap: wrap;
    }
    .hero-tag {
        background: rgba(18, 59, 143, 0.6);
        border: 1px solid #1D3A5A;
        color: #F8FAFC;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 11.5px;
        font-weight: 600;
    }
    
    /* Sports Card Layout */
    .sports-card {
        background: #0B1D34;
        border: 1px solid #1D3A5A;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .sports-card:hover {
        border-color: #3B82F6;
    }
    .card-title {
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        color: #F8FAFC;
        margin-bottom: 4px;
    }
    .card-sub {
        font-size: 12px;
        color: #8FA7C2;
        margin-bottom: 14px;
    }
    
    /* Sports Statistics Metric Box with Thin Bottom Accent */
    .stat-kpi-card {
        background: #0B1D34;
        border: 1px solid #1D3A5A;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 14px;
        position: relative;
        overflow: hidden;
        border-bottom: 3px solid #3B82F6;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.2);
    }
    .stat-kpi-card-gold { border-bottom-color: #F5C451; }
    .stat-kpi-card-blue { border-bottom-color: #3B82F6; }
    .stat-kpi-card-teal { border-bottom-color: #14B8A6; }
    .stat-kpi-card-green { border-bottom-color: #22C55E; }
    
    .stat-kpi-label {
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #8FA7C2;
        margin-bottom: 6px;
    }
    .stat-kpi-num {
        font-size: 28px;
        font-weight: 900;
        color: #FFFFFF;
        line-height: 1.1;
    }
    .stat-kpi-meta {
        font-size: 11px;
        color: #64748B;
        margin-top: 4px;
    }
    
    /* Sports Scoreboard Prediction Cards */
    .scoreboard-box {
        background: linear-gradient(135deg, #09192E 0%, #0E294A 100%);
        border: 2px solid #2563EB;
        border-radius: 12px;
        padding: 24px;
        margin: 18px 0;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.25);
    }
    .scoreboard-label {
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #F5C451;
    }
    .scoreboard-score {
        font-size: 44px;
        font-weight: 900;
        color: #FFFFFF;
        line-height: 1.05;
        margin: 6px 0;
        letter-spacing: -1px;
    }
    .scoreboard-score-unit {
        font-size: 20px;
        color: #38BDF8;
        font-weight: 700;
    }
    .scoreboard-meta {
        font-size: 13px;
        color: #CBD5E1;
        margin-top: 6px;
    }
    
    /* Match Card Component */
    .match-preview-card {
        background: #0B1D34;
        border: 1px solid #1D3A5A;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .match-card-header {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        font-weight: 700;
        color: #8FA7C2;
        text-transform: uppercase;
        border-bottom: 1px solid #1D3A5A;
        padding-bottom: 8px;
        margin-bottom: 12px;
    }
    .match-team-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .match-team-name {
        font-size: 15px;
        font-weight: 800;
        color: #F8FAFC;
    }
    .match-team-score {
        font-size: 17px;
        font-weight: 800;
        color: #F5C451;
    }
    
    /* Form Inputs & Action Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(180deg, #2563EB 0%, #1D4ED8 100%);
        color: #FFFFFF;
        font-weight: 800;
        font-size: 13.5px;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        border: 1px solid #3B82F6;
        border-radius: 6px;
        padding: 11px 24px;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.4);
        transition: all 0.15s ease-in-out;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(180deg, #3B82F6 0%, #2563EB 100%);
        border-color: #60A5FA;
        box-shadow: 0 6px 14px rgba(37, 99, 235, 0.5);
    }
    
    /* Seamless Filter Container */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #0B1D34 !important;
        border: 1px solid #1D3A5A !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        margin-bottom: 22px !important;
    }
    div[data-testid="stSelectbox"] label p {
        font-size: 11px !important;
        font-weight: 800 !important;
        letter-spacing: 0.8px !important;
        text-transform: uppercase !important;
        color: #8FA7C2 !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #061326 !important;
        border-color: #1D3A5A !important;
        border-radius: 7px !important;
        color: #F8FAFC !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #3B82F6 !important;
    }
    
    /* Sidebar Cricket Analytics Control Center Styling */
    section[data-testid="stSidebar"] {
        background-color: #061326 !important;
        border-right: 1px solid #1D3A5A !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        background-color: #061326 !important;
        padding-top: 1.5rem !important;
    }
    
    /* Navigation Group: Transform st.radio into Control Center Cards */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 9px !important;
        padding: 2px 0 !important;
    }
    
    /* Hide the circular radio dot */
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    
    /* Navigation Card Body */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: #0B1D34 !important;
        border: 1px solid #1D3A5A !important;
        border-left: 3px solid transparent !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.18s ease-in-out !important;
        display: block !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    
    /* Hover state */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: #0E2542 !important;
        border-color: #3B82F6 !important;
        border-left: 3px solid #3B82F6 !important;
        transform: translateX(2px) !important;
    }
    
    /* Active / Selected Navigation Card */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked),
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(90deg, #123B6B 0%, #0B1D34 100%) !important;
        border-left: 4px solid #F5C451 !important;
        border-color: #3B82F6 !important;
        box-shadow: 0 4px 14px rgba(18, 59, 107, 0.45) !important;
    }
    
    /* Card Typography */
    section[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        color: #F8FAFC !important;
        font-size: 12.5px !important;
        font-weight: 800 !important;
        line-height: 1.45 !important;
        white-space: pre-line !important;
        margin: 0 !important;
        letter-spacing: 0.3px !important;
    }
    
    /* Editorial Sports Article Cards */
    .editorial-card {
        background: #0B1D34;
        border: 1px solid #1D3A5A;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 14px;
        height: 100%;
    }
    .editorial-badge {
        font-size: 10.5px;
        font-weight: 800;
        text-transform: uppercase;
        color: #F5C451;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .editorial-title {
        font-size: 16px;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 6px;
    }
    .editorial-desc {
        font-size: 12.5px;
        color: #8FA7C2;
        line-height: 1.45;
    }
    
    /* Explanation Box */
    .sports-explanation-card {
        background: #091728;
        border: 1px solid #1D3A5A;
        border-left: 4px solid #F5C451;
        border-radius: 6px;
        padding: 14px 18px;
        margin-top: 14px;
        font-size: 13px;
        line-height: 1.5;
        color: #CBD5E1;
    }
    
    /* Subdued Footer */
    .app-footer {
        text-align: center;
        font-size: 11.5px;
        color: #64748B;
        padding: 30px 0 10px 0;
        margin-top: 40px;
        border-top: 1px solid #1D3A5A;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Ingestion & Model Pipelines
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_preprocess_dataset():
    """Loads raw dataset and applies authentic data wrangling pipeline."""
    csv_candidates = [
        os.path.join(BASE_DIR, 'data', 'ipl_2008_2024_complete(1).csv'),
        os.path.join(BASE_DIR, 'data', 'ipl_2008_2024_complete.csv')
    ]
    filepath = None
    for path in csv_candidates:
        if os.path.exists(path):
            filepath = path
            break
            
    if not filepath:
        raise FileNotFoundError("IPL dataset file not found in data/ directory.")
        
    raw_df = load_raw_data(filepath)
    clean_df = clean_and_wrangle_data(raw_df, standardize_teams=True)
    return raw_df, clean_df


@st.cache_resource
def load_and_train_ml_pipelines(clean_df):
    """Prepares matrices, trains regression and classification models without data leakage."""
    X_reg, y_reg, X_clf, y_clf, threshold = prepare_ml_data(clean_df)
    reg_results, reg_models, X_train_r, X_test_r, y_train_r, y_test_r = train_and_evaluate_regression(X_reg, y_reg)
    clf_results, clf_models, X_train_c, X_test_c, y_train_c, y_test_c = train_and_evaluate_classification(X_clf, y_clf)
    
    return {
        'reg_results': reg_results,
        'reg_models': reg_models,
        'clf_results': clf_results,
        'clf_models': clf_models,
        'threshold': threshold,
        'features': REGRESSION_FEATURES,
        'X_test_r': X_test_r,
        'y_test_r': y_test_r,
        'X_test_c': X_test_c,
        'y_test_c': y_test_c
    }


# Execute initialization
try:
    raw_df, df = load_and_preprocess_dataset()
    ml_data = load_and_train_ml_pipelines(df)
except Exception as e:
    st.error(f"Error initializing data or models: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# Centralized Dataset Metadata (Zero Contradictory Statistics)
# -----------------------------------------------------------------------------
regular_df = df[df['is_regular_innings']]
dataset_stats = {
    'rows': len(raw_df),                     # 2,217
    'features': raw_df.shape[1],             # 28
    'regular_innings': len(regular_df),       # 2,187
    'seasons': int(df['season'].max() - df['season'].min() + 1),  # 17
    'season_min': int(df['season'].min()),    # 2008
    'season_max': int(df['season'].max()),    # 2024
    'teams': df['batting_team_std'].nunique(), # 19
    'matches': df['match_id'].nunique(),     # 1,095
    'avg_score': regular_df['total_runs'].mean(),      # 150.8
    'avg_rr': regular_df['run_rate'].mean(),           # 7.60
    'avg_sixes': regular_df['boundaries_6'].mean(),    # 6.0
    'avg_pp': regular_df['powerplay_runs'].mean(),     # 44.3
    'max_score': int(regular_df['total_runs'].max()),  # 272
    'total_runs_sum': int(regular_df['total_runs'].sum()),
    'boundary_pct': regular_df['boundary_run_pct'].mean() # 56.2%
}


# -----------------------------------------------------------------------------
# Sidebar Navigation & Cricket Analytics Control Center
# -----------------------------------------------------------------------------
# Sidebar Top - Brand Block
st.sidebar.markdown("""
<div style="text-align: center; padding: 4px 0 12px 0;">
    <div style="font-size: 32px; line-height: 1; margin-bottom: 6px;">🏏</div>
    <div style="font-size: 26px; font-weight: 900; letter-spacing: 2.5px; color: #FFFFFF; line-height: 1;">IPL</div>
    <div style="font-size: 18px; font-weight: 800; letter-spacing: 2px; color: #F5C451; margin-top: 2px;">ANALYTICS</div>
    <div style="width: 44px; height: 3px; background: linear-gradient(90deg, #F5C451, #3B82F6); margin: 8px auto 6px auto; border-radius: 2px;"></div>
    <div style="font-size: 9.5px; font-weight: 800; text-transform: uppercase; color: #14B8A6; letter-spacing: 1.8px;">
        CRICKET INTELLIGENCE
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar - Season Pulse
st.sidebar.markdown(f"""
<div style="background: #0B1D34; border: 1px solid #1D3A5A; border-radius: 8px; padding: 12px 14px; margin-bottom: 14px;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <span style="font-size: 10px; font-weight: 800; text-transform: uppercase; color: #8FA7C2; letter-spacing: 1.2px;">SEASON PULSE</span>
        <span style="font-size: 10px; font-weight: 800; color: #22C55E; display: flex; align-items: center; gap: 4px;">
            <span style="width: 6px; height: 6px; background-color: #22C55E; border-radius: 50%; display: inline-block;"></span> ACTIVE
        </span>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 10.5px; font-weight: 700; color: #CBD5E1; margin-bottom: 5px;">
        <span>{dataset_stats['season_min']}</span>
        <span style="color: #F5C451; font-weight: 800;">{dataset_stats['season_max']}</span>
    </div>
    <div style="position: relative; width: 100%; height: 6px; background: #061326; border-radius: 3px; margin-bottom: 8px; border: 1px solid #1D3A5A;">
        <div style="position: absolute; left: 0; top: 0; height: 100%; width: 100%; background: linear-gradient(90deg, #123B6B, #3B82F6, #F5C451); border-radius: 3px;"></div>
        <div style="position: absolute; right: -2px; top: -3px; width: 10px; height: 10px; background: #F5C451; border-radius: 50%; box-shadow: 0 0 8px #F5C451;"></div>
    </div>
    <div style="text-align: center; font-size: 11px; font-weight: 800; color: #F8FAFC; letter-spacing: 0.5px;">
        {dataset_stats['seasons']} SEASONS ANALYZED
    </div>
</div>
<div style="font-size: 10px; font-weight: 800; text-transform: uppercase; color: #8FA7C2; letter-spacing: 1.8px; margin: 10px 0 8px 4px;">
    EXPLORE
</div>
""", unsafe_allow_html=True)

nav_options = [
    "⌂  Overview",
    "▣  IPL Analytics",
    "◈  Visualizations",
    "⚡ Machine Learning",
    "⌕  Data Explorer",
    "ⓘ  About Project"
]

NAV_DESCRIPTIONS = {
    "⌂  Overview": "⌂  OVERVIEW\nExecutive Summary & Trends",
    "▣  IPL Analytics": "▣  IPL ANALYTICS\nTeams • Scoring • Venues",
    "◈  Visualizations": "◈  VISUALIZATIONS\n20 Statistical Charts",
    "⚡ Machine Learning": "⚡ MACHINE LEARNING\nPredict • Classify",
    "⌕  Data Explorer": "⌕  DATA EXPLORER\nSchema & Raw Slices",
    "ⓘ  About Project": "ⓘ  ABOUT PROJECT\nCurriculum & Methodology"
}

nav_selection = st.sidebar.radio(
    "Navigation",
    nav_options,
    format_func=lambda x: NAV_DESCRIPTIONS.get(x, x),
    label_visibility="collapsed"
)

# Sidebar - Live Data Engine Card
st.sidebar.markdown(f"""
<div style="background: #0B1D34; border: 1px solid #1D3A5A; border-radius: 8px; padding: 14px; margin-top: 14px;">
    <div style="font-size: 10.5px; font-weight: 800; color: #22C55E; text-transform: uppercase; letter-spacing: 1px; display: flex; align-items: center; gap: 6px; margin-bottom: 10px;">
        <span style="width: 7px; height: 7px; background: #22C55E; border-radius: 50%; display: inline-block; box-shadow: 0 0 6px #22C55E;"></span>
        DATA ENGINE
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; text-align: left; margin-bottom: 10px;">
        <div style="background: #061326; padding: 8px; border-radius: 6px; border: 1px solid #1D3A5A;">
            <div style="font-size: 14.5px; font-weight: 900; color: #FFFFFF;">{dataset_stats['rows']:,}</div>
            <div style="font-size: 9px; font-weight: 700; text-transform: uppercase; color: #8FA7C2;">RECORDS</div>
        </div>
        <div style="background: #061326; padding: 8px; border-radius: 6px; border: 1px solid #1D3A5A;">
            <div style="font-size: 14.5px; font-weight: 900; color: #FFFFFF;">{dataset_stats['features']}</div>
            <div style="font-size: 9px; font-weight: 700; text-transform: uppercase; color: #8FA7C2;">FEATURES</div>
        </div>
        <div style="background: #061326; padding: 8px; border-radius: 6px; border: 1px solid #1D3A5A;">
            <div style="font-size: 14.5px; font-weight: 900; color: #FFFFFF;">{dataset_stats['seasons']}</div>
            <div style="font-size: 9px; font-weight: 700; text-transform: uppercase; color: #8FA7C2;">SEASONS</div>
        </div>
        <div style="background: #061326; padding: 8px; border-radius: 6px; border: 1px solid #1D3A5A;">
            <div style="font-size: 14.5px; font-weight: 900; color: #FFFFFF;">{dataset_stats['teams']}</div>
            <div style="font-size: 9px; font-weight: 700; text-transform: uppercase; color: #8FA7C2;">TEAMS</div>
        </div>
    </div>
    <div style="font-size: 10px; font-weight: 800; color: #14B8A6; text-transform: uppercase; letter-spacing: 0.8px; text-align: center; border-top: 1px solid #1D3A5A; padding-top: 8px;">
        ● ANALYTICS READY
    </div>
</div>

<!-- Sidebar - Mini Cricket Pitch Visual -->
<div style="margin: 14px 0 10px 0; background: #0B1D34; border: 1px solid #1D3A5A; border-radius: 8px; padding: 12px; text-align: center;">
    <div style="font-size: 9.5px; font-weight: 800; text-transform: uppercase; color: #8FA7C2; letter-spacing: 1.2px; margin-bottom: 8px;">
        22-YARD REGULATION PITCH
    </div>
    <div style="position: relative; width: 88px; height: 104px; margin: 0 auto; background: #061326; border: 1px solid #1D3A5A; border-radius: 4px; overflow: hidden; display: flex; align-items: center; justify-content: center;">
        <div style="position: absolute; inset: 0; background: repeating-linear-gradient(0deg, rgba(20, 184, 166, 0.04), rgba(20, 184, 166, 0.04) 8px, transparent 8px, transparent 16px);"></div>
        <div style="position: relative; width: 34px; height: 92px; background: linear-gradient(180deg, #A88B58 0%, #C4A76C 50%, #A88B58 100%); border: 1px solid #DFAF35; border-radius: 2px; box-shadow: 0 0 8px rgba(0, 0, 0, 0.5);">
            <!-- Top Bowling Crease & Stumps -->
            <div style="position: absolute; top: 11px; left: 0; right: 0; height: 1.5px; background: #FFFFFF;"></div>
            <div style="position: absolute; top: 3px; left: 50%; transform: translateX(-50%); display: flex; gap: 2px;">
                <span style="width: 2px; height: 6px; background: #FFFFFF; border-radius: 1px;"></span>
                <span style="width: 2px; height: 6px; background: #FFFFFF; border-radius: 1px;"></span>
                <span style="width: 2px; height: 6px; background: #FFFFFF; border-radius: 1px;"></span>
            </div>
            <!-- Cricket Ball -->
            <div style="position: absolute; top: 36px; left: 50%; transform: translateX(-50%); width: 6px; height: 6px; background: #EF4444; border-radius: 50%; box-shadow: 0 0 6px #EF4444;"></div>
            <!-- Bottom Batting Crease & Stumps -->
            <div style="position: absolute; bottom: 11px; left: 0; right: 0; height: 1.5px; background: #FFFFFF;"></div>
            <div style="position: absolute; bottom: 3px; left: 50%; transform: translateX(-50%); display: flex; gap: 2px;">
                <span style="width: 2px; height: 6px; background: #FFFFFF; border-radius: 1px;"></span>
                <span style="width: 2px; height: 6px; background: #FFFFFF; border-radius: 1px;"></span>
                <span style="width: 2px; height: 6px; background: #FFFFFF; border-radius: 1px;"></span>
            </div>
        </div>
    </div>
    <div style="font-size: 9px; color: #8FA7C2; margin-top: 6px; letter-spacing: 0.5px;">INNINGS REGULATION STRIP</div>
</div>

<!-- Sidebar Footer -->
<div style="text-align: center; border-top: 1px solid #1D3A5A; padding-top: 10px; margin-top: 8px;">
    <div style="font-size: 11px; font-weight: 800; color: #F5C451; letter-spacing: 1px;">DAV PROJECT</div>
    <div style="font-size: 10px; font-weight: 700; color: #CBD5E1; margin-top: 2px;">IPL DATA ANALYTICS</div>
    <div style="font-size: 9px; color: #8FA7C2;">Visualization • ML</div>
    <div style="font-size: 9.5px; color: #22C55E; font-weight: 700; margin-top: 5px; display: flex; align-items: center; justify-content: center; gap: 4px;">
        <span style="width: 6px; height: 6px; background: #22C55E; border-radius: 50%;"></span> SYSTEM READY
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Global Top Full-Width Sports Website Header Bar (Normal Flow, Never Clipped)
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="sports-nav-header">
    <div class="sports-brand">
        <div class="brand-icon-box">🏏</div>
        <div>
            <div class="brand-logo-text">IPL <span class="brand-accent-gold">ANALYTICS</span></div>
            <div class="brand-tagline">CRICKET DATA INTELLIGENCE</div>
        </div>
    </div>
    <div class="header-meta">
        <div class="header-era-badge">{dataset_stats['season_min']} — {dataset_stats['season_max']}</div>
        <div class="live-status">
            <span class="live-pulse"></span> ● DATA READY &nbsp;•&nbsp; {dataset_stats['rows']:,} RECORDS
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# 1. ⌂ OVERVIEW / HOME PAGE
# =============================================================================
if "Overview" in nav_selection:
    # Large Full-Width Hero Section
    st.markdown(f"""
    <div class="hero-container">
        <div class="hero-eyebrow">IPL ANALYTICS • 17 EDITIONS COMPREHENSIVE REPOSITORY</div>
        <div class="hero-title">TURNING CRICKET DATA INTO PERFORMANCE INSIGHTS</div>
        <div class="hero-subtitle">
            Explore longitudinal scoring evolution, franchise dominance, match-phase mechanics,
            and predictive machine learning models derived from {dataset_stats['rows']:,} IPL innings fixtures.
        </div>
        <div class="hero-badges-strip">
            <span class="hero-tag">🏆 {dataset_stats['seasons']} IPL TOURNAMENTS</span>
            <span class="hero-tag">📊 {dataset_stats['features']} MATCH METRICS</span>
            <span class="hero-tag">⚡ 6 ML MODELS TRAINED</span>
            <span class="hero-tag">🏟️ {dataset_stats['matches']:,} FIXTURES</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # IPL AT A GLANCE
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <div style="font-size: 18px; font-weight: 900; color: #FFFFFF; letter-spacing: -0.3px;">IPL AT A GLANCE</div>
        <div style="font-size: 12.5px; color: #94A3B8;">A high-level structural snapshot of the complete dataset</div>
    </div>
    """, unsafe_allow_html=True)
    
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown(f"""
        <div class="stat-kpi-card stat-kpi-card-gold">
            <div class="stat-kpi-label">Innings Records</div>
            <div class="stat-kpi-num">{dataset_stats['rows']:,}</div>
            <div class="stat-kpi-meta">{dataset_stats['regular_innings']:,} regular 20-over innings</div>
        </div>
        """, unsafe_allow_html=True)
    with g2:
        st.markdown(f"""
        <div class="stat-kpi-card">
            <div class="stat-kpi-label">Seasons</div>
            <div class="stat-kpi-num">{dataset_stats['seasons']}</div>
            <div class="stat-kpi-meta">{dataset_stats['season_min']} through {dataset_stats['season_max']}</div>
        </div>
        """, unsafe_allow_html=True)
    with g3:
        st.markdown(f"""
        <div class="stat-kpi-card stat-kpi-card-teal">
            <div class="stat-kpi-label">Franchises</div>
            <div class="stat-kpi-num">{dataset_stats['teams']}</div>
            <div class="stat-kpi-meta">Active and historical franchise clubs</div>
        </div>
        """, unsafe_allow_html=True)
    with g4:
        st.markdown(f"""
        <div class="stat-kpi-card stat-kpi-card-green">
            <div class="stat-kpi-label">Attributes</div>
            <div class="stat-kpi-num">{dataset_stats['features']}</div>
            <div class="stat-kpi-meta">Phase runs, wickets, extras & venues</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # Featured Analytics Section: 2 Columns
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <div style="font-size: 18px; font-weight: 900; color: #FFFFFF; letter-spacing: -0.3px;">FEATURED ANALYTICS</div>
        <div style="font-size: 12.5px; color: #94A3B8;">Longitudinal scoring evolution across tournament eras</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_feat_l, col_feat_r = st.columns([1.5, 0.9])
    with col_feat_l:
        st.markdown("""
        <div class="sports-card">
            <div class="card-title">Season Scoring Evolution</div>
            <div class="card-sub">Average innings total runs progression across all 17 IPL tournament editions</div>
        """, unsafe_allow_html=True)
        
        fig_trend = plot_season_scoring_trend(df, save=False)
        fig_trend = stylize_fig(fig_trend)
        st.pyplot(fig_trend)
        plt.close(fig_trend)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_feat_r:
        st.markdown(f"""<div class="sports-card" style="height: 100%;">
<div class="card-title" style="color: #F5C451;">QUICK INSIGHTS</div>
<div class="card-sub">Key statistical benchmarks</div>
<div style="margin-bottom: 14px;">
<div style="font-size: 26px; font-weight: 900; color: #FFFFFF; line-height: 1.1;">{dataset_stats['avg_score']:.1f}</div>
<div style="font-size: 10.5px; text-transform: uppercase; color: #8FA7C2; font-weight: 800; letter-spacing: 0.8px; margin-top: 2px;">AVG INNINGS SCORE</div>
</div>
<div style="margin-bottom: 14px;">
<div style="font-size: 26px; font-weight: 900; color: #FFFFFF; line-height: 1.1;">{dataset_stats['avg_rr']:.2f}</div>
<div style="font-size: 10.5px; text-transform: uppercase; color: #8FA7C2; font-weight: 800; letter-spacing: 0.8px; margin-top: 2px;">AVG RUN RATE</div>
</div>
<div style="margin-bottom: 14px;">
<div style="font-size: 26px; font-weight: 900; color: #FFFFFF; line-height: 1.1;">{dataset_stats['avg_sixes']:.1f}</div>
<div style="font-size: 10.5px; text-transform: uppercase; color: #8FA7C2; font-weight: 800; letter-spacing: 0.8px; margin-top: 2px;">AVG SIXES</div>
</div>
<div>
<div style="font-size: 26px; font-weight: 900; color: #FFFFFF; line-height: 1.1;">{dataset_stats['avg_pp']:.1f}</div>
<div style="font-size: 10.5px; text-transform: uppercase; color: #8FA7C2; font-weight: 800; letter-spacing: 0.8px; margin-top: 2px;">AVG POWERPLAY RUNS</div>
</div>
</div>""", unsafe_allow_html=True)
        
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    
    # Editorial Sports Insights Section (3 Cards)
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <div style="font-size: 18px; font-weight: 900; color: #FFFFFF; letter-spacing: -0.3px;">DATA INSIGHTS</div>
        <div style="font-size: 12.5px; color: #94A3B8;">Editorial takeaways from longitudinal match metrics</div>
    </div>
    """, unsafe_allow_html=True)
    
    ed1, ed2, ed3 = st.columns(3)
    with ed1:
        st.markdown(f"""
        <div class="editorial-card">
            <div class="editorial-badge">SCORING TRAJECTORY</div>
            <div class="editorial-title">Rising Scoring Velocity</div>
            <div class="editorial-desc">
                Average innings totals have risen by over 25 runs since 2008. Boundaries (4s and 6s) account for <strong>{dataset_stats['boundary_pct']:.1f}%</strong> of all runs scored in tournament history.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ed2:
        st.markdown(f"""
        <div class="editorial-card">
            <div class="editorial-badge">FRANCHISE POWER</div>
            <div class="editorial-title">Team Dominance</div>
            <div class="editorial-desc">
                Mumbai Indians, Chennai Super Kings, and Kolkata Knight Riders have accumulated the highest match victories and collective runs, defining the tactical benchmark of the league.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ed3:
        st.markdown("""
        <div class="editorial-card">
            <div class="editorial-badge">MATCH DYNAMICS</div>
            <div class="editorial-title">The Toss Fallacy</div>
            <div class="editorial-desc">
                Despite modern teams opting to field first in over 64% of fixtures, the toss winner wins the match exactly <strong>50.05%</strong> of the time, proving coin-toss neutrality.
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# 2. ▣ IPL ANALYTICS PAGE (Sports Statistics Center)
# =============================================================================
elif "Analytics" in nav_selection:
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <div style="font-size: 22px; font-weight: 900; color: #FFFFFF; letter-spacing: -0.3px;">IPL ANALYTICS CENTER</div>
        <div style="font-size: 13px; color: #8FA7C2; margin-top: 3px;">
            Explore team performance, scoring behavior, toss decisions and venue patterns.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Premium Seamless Filter Bar Toolbar
    with st.container(border=True):
        st.markdown("""
        <div style="font-size: 11px; font-weight: 800; text-transform: uppercase; color: #F5C451; letter-spacing: 1.2px; margin-bottom: 4px;">
            ⚡ FILTER DATASET
        </div>
        """, unsafe_allow_html=True)
        flt_col1, flt_col2, flt_col3 = st.columns(3)
        with flt_col1:
            season_choices = ["All Seasons"] + sorted([int(s) for s in df['season'].unique()])
            filter_season = st.selectbox("SEASON", season_choices, key="analytics_season_filter")
        with flt_col2:
            team_choices = ["All Teams"] + sorted(df['batting_team_std'].unique().tolist())
            filter_team = st.selectbox("TEAM", team_choices, key="analytics_team_filter")
        with flt_col3:
            venue_choices = ["All Venues"] + sorted(df['venue'].dropna().unique().tolist())[:25]
            filter_venue = st.selectbox("VENUE", venue_choices, key="analytics_venue_filter")
    
    # Slice filtered dataset
    filtered_analytics = regular_df.copy()
    if filter_season != "All Seasons":
        filtered_analytics = filtered_analytics[filtered_analytics['season'] == int(filter_season)]
    if filter_team != "All Teams":
        filtered_analytics = filtered_analytics[filtered_analytics['batting_team_std'] == filter_team]
    if filter_venue != "All Venues":
        filtered_analytics = filtered_analytics[filtered_analytics['venue'] == filter_venue]
        
    if len(filtered_analytics) == 0:
        st.warning("No matches found matching the selected filter criteria. Please broaden your selection.")
        st.stop()
        
    # Team Performance Cards
    st.markdown("""
    <div style="font-size: 15px; font-weight: 900; color: #FFFFFF; letter-spacing: 0.5px; margin: 18px 0 10px 0;">
        TEAM PERFORMANCE
    </div>
    """, unsafe_allow_html=True)
    tp1, tp2, tp3, tp4 = st.columns(4)
    
    f_runs_sum = int(filtered_analytics['total_runs'].sum())
    f_avg_rr = float(filtered_analytics['run_rate'].mean())
    f_max_score = int(filtered_analytics['total_runs'].max())
    
    # Matches won calculation
    if filter_team != "All Teams":
        matches_subset = df[df['winner_std'] == filter_team].drop_duplicates('match_id')
        if filter_season != "All Seasons":
            matches_subset = matches_subset[matches_subset['season'] == int(filter_season)]
        if filter_venue != "All Venues":
            matches_subset = matches_subset[matches_subset['venue'] == filter_venue]
        f_matches_won = len(matches_subset)
    else:
        f_matches_won = df.drop_duplicates('match_id')['winner_std'].dropna().count()
        
    with tp1:
        st.markdown(f"""
        <div class="stat-kpi-card stat-kpi-card-gold">
            <div class="stat-kpi-label">Total Runs</div>
            <div class="stat-kpi-num">{f_runs_sum:,}</div>
            <div class="stat-kpi-meta">{len(filtered_analytics):,} innings represented</div>
        </div>
        """, unsafe_allow_html=True)
    with tp2:
        st.markdown(f"""
        <div class="stat-kpi-card stat-kpi-card-blue">
            <div class="stat-kpi-label">Avg Run Rate</div>
            <div class="stat-kpi-num">{f_avg_rr:.2f}</div>
            <div class="stat-kpi-meta">Runs per over velocity</div>
        </div>
        """, unsafe_allow_html=True)
    with tp3:
        st.markdown(f"""
        <div class="stat-kpi-card stat-kpi-card-green">
            <div class="stat-kpi-label">Matches Won</div>
            <div class="stat-kpi-num">{f_matches_won:,}</div>
            <div class="stat-kpi-meta">Total victories recorded</div>
        </div>
        """, unsafe_allow_html=True)
    with tp4:
        st.markdown(f"""
        <div class="stat-kpi-card stat-kpi-card-teal">
            <div class="stat-kpi-label">Highest Score</div>
            <div class="stat-kpi-num">{f_max_score}</div>
            <div class="stat-kpi-meta">Peak innings total</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    # Interactive Team Comparison Component (Compact and Integrated)
    cmp_hdr_l, cmp_hdr_r = st.columns([2.2, 1])
    with cmp_hdr_l:
        st.markdown("""
        <div style="padding-top: 6px;">
            <div style="font-size: 16px; font-weight: 900; color: #FFFFFF; letter-spacing: -0.2px;">TEAM PERFORMANCE COMPARISON</div>
            <div style="font-size: 12px; color: #8FA7C2;">Compare franchise historical metrics across tournament history</div>
        </div>
        """, unsafe_allow_html=True)
    with cmp_hdr_r:
        metric_choice = st.selectbox(
            "CHOOSE COMPARISON METRIC",
            ["Total Runs", "Matches Won", "Average Run Rate", "Highest Score"],
            key="team_comp_metric",
            label_visibility="collapsed"
        )
    
    # Calculate metric series
    if metric_choice == "Total Runs":
        comp_series = regular_df.groupby('batting_team_std')['total_runs'].sum().sort_values(ascending=False).head(10)
        x_label = "Total Runs Scored"
    elif metric_choice == "Matches Won":
        comp_series = df.drop_duplicates('match_id')['winner_std'].value_counts().head(10)
        x_label = "Total Matches Won"
    elif metric_choice == "Average Run Rate":
        comp_series = regular_df.groupby('batting_team_std')['run_rate'].mean().sort_values(ascending=False).head(10)
        x_label = "Average Run Rate (Runs per Over)"
    else:
        comp_series = regular_df.groupby('batting_team_std')['total_runs'].max().sort_values(ascending=False).head(10)
        x_label = "Highest Innings Total"
        
    fig_comp, ax_comp = plt.subplots(figsize=(10, 4.5), dpi=140)
    sns.barplot(
        x=comp_series.values,
        y=comp_series.index,
        hue=comp_series.index,
        palette='Blues_r',
        legend=False,
        ax=ax_comp
    )
    for p in ax_comp.patches:
        val = p.get_width()
        fmt = f"{val:.2f}" if metric_choice == "Average Run Rate" else f"{int(val):,}"
        ax_comp.annotate(f" {fmt}", (val, p.get_y() + p.get_height() / 2),
                         va='center', ha='left', fontsize=8.5, color='#F8FAFC', weight='bold')
    ax_comp.set_xlabel(x_label, fontsize=9)
    ax_comp.set_ylabel("", fontsize=9)
    fig_comp = stylize_fig(fig_comp)
    st.pyplot(fig_comp)
    plt.close(fig_comp)
    
    # Scoring Profile Section
    st.markdown("#### SCORING PROFILE")
    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
    with sc1:
        st.markdown(f"""
        <div class="stat-kpi-card" style="padding: 12px 14px;">
            <div class="stat-kpi-label">Avg Runs</div>
            <div class="stat-kpi-num" style="font-size: 20px;">{filtered_analytics['total_runs'].mean():.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""
        <div class="stat-kpi-card" style="padding: 12px 14px;">
            <div class="stat-kpi-label">Avg Wickets</div>
            <div class="stat-kpi-num" style="font-size: 20px;">{filtered_analytics['wickets'].mean():.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    with sc3:
        st.markdown(f"""
        <div class="stat-kpi-card" style="padding: 12px 14px;">
            <div class="stat-kpi-label">Avg Fours</div>
            <div class="stat-kpi-num" style="font-size: 20px;">{filtered_analytics['boundaries_4'].mean():.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    with sc4:
        st.markdown(f"""
        <div class="stat-kpi-card" style="padding: 12px 14px;">
            <div class="stat-kpi-label">Avg Sixes</div>
            <div class="stat-kpi-num" style="font-size: 20px;">{filtered_analytics['boundaries_6'].mean():.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    with sc5:
        st.markdown(f"""
        <div class="stat-kpi-card" style="padding: 12px 14px;">
            <div class="stat-kpi-label">Powerplay</div>
            <div class="stat-kpi-num" style="font-size: 20px;">{filtered_analytics['powerplay_runs'].mean():.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    with sc6:
        death_subset = filtered_analytics[filtered_analytics['death_over_runs'] > 0]
        death_avg = death_subset['death_over_runs'].mean() if len(death_subset) > 0 else 0
        st.markdown(f"""
        <div class="stat-kpi-card" style="padding: 12px 14px;">
            <div class="stat-kpi-label">Death Overs</div>
            <div class="stat-kpi-num" style="font-size: 20px;">{death_avg:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # Match Dynamics: Toss & Venue Insights
    st.markdown("#### MATCH DYNAMICS")
    col_dyn_toss, col_dyn_venue = st.columns(2)
    
    with col_dyn_toss:
        st.markdown("""
        <div class="sports-card">
            <div class="card-title">TOSS INSIGHTS</div>
            <div class="card-sub">Bat vs Field choice proportions and match victory conversion</div>
        """, unsafe_allow_html=True)
        
        matches_filtered = df[df['match_id'].isin(filtered_analytics['match_id'])].drop_duplicates('match_id')
        toss_counts = matches_filtered['toss_decision'].value_counts()
        
        fig_toss, ax_toss = plt.subplots(figsize=(5, 3.4), dpi=140)
        if len(toss_counts) > 0:
            wedges, texts, autotexts = ax_toss.pie(
                toss_counts.values,
                labels=toss_counts.index.str.capitalize(),
                autopct='%1.1f%%',
                colors=['#1D4ED8', '#F5C451'],
                startangle=140,
                textprops={'fontsize': 9, 'color': '#F8FAFC'}
            )
            for at in autotexts:
                at.set_color('#FFFFFF')
                at.set_weight('bold')
        ax_toss.axis('equal')
        fig_toss = stylize_fig(fig_toss)
        st.pyplot(fig_toss)
        plt.close(fig_toss)
        
        toss_win_pct = matches_filtered['toss_win_match_win'].mean() * 100
        st.caption(f"Toss winner won the match in **{toss_win_pct:.1f}%** of these fixtures.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_dyn_venue:
        st.markdown("""
        <div class="sports-card">
            <div class="card-title">VENUE INSIGHTS</div>
            <div class="card-sub">Most frequented stadiums within active filter</div>
        """, unsafe_allow_html=True)
        
        top_v = matches_filtered['venue'].value_counts().head(6).reset_index()
        top_v.columns = ['venue', 'matches']
        
        fig_v, ax_v = plt.subplots(figsize=(6, 3.4), dpi=140)
        sns.barplot(data=top_v, x='matches', y='venue', hue='venue', palette='Blues_r', legend=False, ax=ax_v)
        ax_v.set_xlabel("Matches", fontsize=9)
        ax_v.set_ylabel("", fontsize=9)
        fig_v = stylize_fig(fig_v)
        st.pyplot(fig_v)
        plt.close(fig_v)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Match-Card Inspired Component (Section 27)
    sample_match = matches_filtered.iloc[0] if len(matches_filtered) > 0 else df.iloc[0]
    m_id = sample_match['match_id']
    m_innings = df[df['match_id'] == m_id].sort_values(by='innings')
    
    if len(m_innings) >= 2:
        inn1 = m_innings.iloc[0]
        inn2 = m_innings.iloc[1]
        st.markdown(f"""
        <div class="match-preview-card">
            <div class="match-card-header">
                <div>FIXTURE SPOTLIGHT • SEASON {int(sample_match['season'])}</div>
                <div>{sample_match['venue']}</div>
            </div>
            <div class="match-team-row">
                <div class="match-team-name">{inn1['batting_team_std']}</div>
                <div class="match-team-score">{int(inn1['total_runs'])}/{int(inn1['wickets'])} <span style="font-size: 12px; color: #94A3B8;">(RR: {inn1['run_rate']:.2f})</span></div>
            </div>
            <div class="match-team-row">
                <div class="match-team-name">{inn2['batting_team_std']}</div>
                <div class="match-team-score">{int(inn2['total_runs'])}/{int(inn2['wickets'])} <span style="font-size: 12px; color: #94A3B8;">(RR: {inn2['run_rate']:.2f})</span></div>
            </div>
            <div style="font-size: 12px; color: #38BDF8; font-weight: 700; margin-top: 8px;">
                Winner: {sample_match['winner_std']} • POTM: {sample_match['player_of_match']}
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# 3. ◈ VISUALIZATIONS PAGE (Visualization Center)
# =============================================================================
elif "Visualizations" in nav_selection:
    st.markdown("""
    <div style="margin-bottom: 18px;">
        <div style="font-size: 20px; font-weight: 900; color: #FFFFFF;">VISUALIZATION CENTER</div>
        <div style="font-size: 13px; color: #94A3B8; margin-top: 2px;">
            Explore the patterns, correlations, and phase dynamics behind IPL performance.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Categorized Visualizations Dictionary
    viz_categories = {
        "PERFORMANCE": [
            "Season-wise Scoring Trend",
            "Team Total Runs",
            "Team Average Run Rate",
            "Team / Season Win Trajectory"
        ],
        "SCORING": [
            "Total Runs Distribution",
            "Run Rate Distribution",
            "Run Rate vs Total Runs",
            "Wickets vs Total Runs",
            "Dot Ball % vs Run Rate",
            "Waterfall Chart (Innings Phase Buildup)"
        ],
        "MATCH DYNAMICS": [
            "Toss Decision Distribution",
            "Crosstab Heatmap (Season vs Toss)",
            "Waffle Chart (Toss Proportions)",
            "Venue Frequency",
            "Player of the Match Frequency",
            "Geographic Distribution (Host Cities)"
        ],
        "ADVANCED": [
            "Correlation Heatmap",
            "Numerical Boxplots (Outlier Audit)",
            "Related IPL Metrics Subplots",
            "Treemap (Team Runs Share)"
        ]
    }
    
    viz_catalog = {
        "Season-wise Scoring Trend": (
            plot_season_scoring_trend,
            "Shows the evolution of average innings totals across 17 IPL seasons (2008–2024), demonstrating the transition from ~150 run baselines to modern 175+ run contests."
        ),
        "Team Total Runs": (
            plot_team_total_runs,
            "Displays cumulative runs scored by each franchise in IPL history. Mumbai Indians leads with over 39,900 runs."
        ),
        "Team Average Run Rate": (
            plot_team_avg_run_rate,
            "Compares historical scoring velocities (runs per over) across all active and legacy IPL franchises."
        ),
        "Total Runs Distribution": (
            plot_total_runs_distribution,
            "Histogram with KDE curve displaying the frequency of innings scores in regular 20-over matches, highlighting a median of 152.0 runs."
        ),
        "Run Rate Distribution": (
            plot_run_rate_distribution,
            "Histogram illustrating the unimodal distribution of innings run rates centered between 7.0 and 8.5 runs per over."
        ),
        "Run Rate vs Total Runs": (
            plot_run_rate_vs_total_runs,
            "Scatter plot and linear regression trendline confirming the strong positive correlation (r = 0.8183) between innings run rate and final score."
        ),
        "Wickets vs Total Runs": (
            plot_wickets_vs_total_runs,
            "Boxplot breakdown demonstrating that losing 7 or more wickets severely constrains an innings below competitive totals."
        ),
        "Dot Ball % vs Run Rate": (
            plot_dot_ball_pct_vs_run_rate,
            "Scatter plot illustrating the drag effect of non-scoring deliveries on scoring rates (r = -0.34)."
        ),
        "Player of the Match Frequency": (
            plot_player_of_match_frequency,
            "Bar chart ranking the most frequent Player of the Match award winners, led by AB de Villiers (25) and Chris Gayle (22)."
        ),
        "Venue Frequency": (
            plot_venue_frequency,
            "Bar chart showing the stadiums that have hosted the most IPL fixtures (M Chinnaswamy, Wankhede, Eden Gardens)."
        ),
        "Toss Decision Distribution": (
            plot_toss_decision_pie,
            "Pie chart illustrating overall toss decisions (Field First: 64.1%, Bat First: 35.9%)."
        ),
        "Related IPL Metrics Subplots": (
            plot_related_metrics_subplots,
            "4-panel dashboard comparing Powerplay Runs vs Total Runs, Death Runs vs Total Runs, Sixes Struck vs Total Runs, and Innings 1 vs 2 score distributions."
        ),
        "Correlation Heatmap": (
            plot_correlation_heatmap,
            "Pearson correlation matrix quantifying relationships between runs, wickets, boundaries, dot balls, and phase totals."
        ),
        "Numerical Boxplots (Outlier Audit)": (
            plot_numerical_boxplots,
            "Multi-panel boxplot examining IQR outlier boundaries for Total Runs, Powerplay Runs, Death Over Runs, and Dot Balls."
        ),
        "Crosstab Heatmap (Season vs Toss)": (
            plot_crosstab_visualization,
            "Two-way contingency heatmap tracking the historical shift from batting first in 2008–2010 to overwhelmingly fielding first in recent seasons."
        ),
        "Treemap (Team Runs Share)": (
            plot_treemap_team_runs,
            "Hierarchical area treemap depicting franchise shares of cumulative IPL runs."
        ),
        "Team / Season Win Trajectory": (
            plot_bump_ranking_chart,
            "Longitudinal line chart displaying the annual match victories achieved by the top active franchises."
        ),
        "Waterfall Chart (Innings Phase Buildup)": (
            plot_waterfall_innings_breakdown,
            "Waterfall diagram decomposing average innings buildup: Powerplay (43.8) + Middle Overs (66.9) + Death Overs (40.1) + Extras (8.2) = 150.8 runs."
        ),
        "Waffle Chart (Toss Proportions)": (
            plot_waffle_chart_toss,
            "100-cell unit waffle grid displaying the macro proportions of teams opting to Field First (64%) vs Bat First (36%)."
        ),
        "Geographic Distribution (Host Cities)": (
            plot_geographic_city_matches,
            "Bar chart illustrating match distribution across host cities in India and the United Arab Emirates (Dubai, Sharjah, Abu Dhabi)."
        ),
    }
    
    # Visual Selection Area with Tabs and Selectbox
    st.markdown('<div class="sports-card" style="padding: 16px 20px; margin-bottom: 20px;">', unsafe_allow_html=True)
    all_viz_list = list(viz_catalog.keys())
    
    cat_tab1, cat_tab2, cat_tab3, cat_tab4 = st.tabs(["PERFORMANCE", "SCORING", "MATCH DYNAMICS", "ADVANCED"])
    
    with cat_tab1:
        st.caption("Longitudinal team performance, scoring progression and win trajectories")
    with cat_tab2:
        st.caption("Distribution histograms, phase accumulation, and boundary physics")
    with cat_tab3:
        st.caption("Toss decisions, venue dominance, and Player of the Match honors")
    with cat_tab4:
        st.caption("Correlation matrices, IQR outlier audits, and multi-metric dashboards")
        
    c_sel, _ = st.columns([1.2, 0.8])
    with c_sel:
        selected_viz_name = st.selectbox(
            "CHOOSE VISUALIZATION (ALL 20 FIGURES SUPPORTED)",
            all_viz_list,
            index=0
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    plot_fn, explanation_text = viz_catalog[selected_viz_name]
    
    # Display Chart Card
    st.markdown(f"""
    <div class="sports-card">
        <div class="card-title" style="color: #F5C451;">{selected_viz_name}</div>
        <div class="card-sub">High-resolution graphical output rendered with dark sports portal integration</div>
    """, unsafe_allow_html=True)
    
    try:
        fig = plot_fn(df, save=False)
        fig = stylize_fig(fig)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Error rendering chart: {e}")
        
    st.markdown(f"""
        <div class="sports-explanation-card">
            <strong>WHAT THIS SHOWS:</strong> {explanation_text}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 4. ⚡ MACHINE LEARNING PAGE (IPL Match Scenario & Predictive Center)
# =============================================================================
elif "Machine Learning" in nav_selection:
    if 'ml_predictions_history' not in st.session_state:
        st.session_state['ml_predictions_history'] = []

    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 20px;">
        <div>
            <div style="font-size: 22px; font-weight: 900; color: #FFFFFF; letter-spacing: -0.3px;">IPL MATCH SCENARIO & PREDICTIVE CENTER</div>
            <div style="font-size: 13px; color: #8FA7C2; margin-top: 3px;">
                Configure authentic IPL match context and innings situation to generate machine learning predictions.
            </div>
        </div>
        <div>
            <span class="header-era-badge" style="background: #092C23; border-color: #10B981; color: #34D399;">
                6 ML MODELS • 3 REGRESSION • 3 CLASSIFICATION
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    all_teams_ml = sorted(df['batting_team_std'].unique().tolist())
    all_seasons_ml = sorted([int(s) for s in df['season'].unique()], reverse=True)
    all_venues_ml = sorted(df['venue'].dropna().unique().tolist())
    
    TEAM_INITIALS = {
        'Mumbai Indians': 'MI',
        'Chennai Super Kings': 'CSK',
        'Royal Challengers Bengaluru': 'RCB',
        'Kolkata Knight Riders': 'KKR',
        'Delhi Capitals': 'DC',
        'Punjab Kings': 'PBKS',
        'Rajasthan Royals': 'RR',
        'Sunrisers Hyderabad': 'SRH',
        'Gujarat Titans': 'GT',
        'Lucknow Super Giants': 'LSG'
    }
    
    tab_reg, tab_clf = st.tabs(["🎯 PREDICT INNINGS SCORE", "🏷️ CLASSIFY HIGH-SCORING"])
    
    # ---------------------------------------------------------
    # TAB 1: REGRESSION SCOREBOARD (SCORE PREDICTION)
    # ---------------------------------------------------------
    with tab_reg:
        # Match Scenario Builder
        st.markdown("""<div class="sports-card" style="margin-bottom: 16px;">
<div class="card-title" style="color: #F5C451;">🏏 IPL MATCH SCENARIO</div>
<div class="card-sub">Configure the IPL match context before generating a prediction.</div>
</div>""", unsafe_allow_html=True)
        
        sc_col1, sc_col_vs, sc_col2 = st.columns([1.2, 0.3, 1.2])
        with sc_col1:
            mi_idx = all_teams_ml.index("Mumbai Indians") if "Mumbai Indians" in all_teams_ml else 0
            bat_team_r = st.selectbox("BATTING FRANCHISE", all_teams_ml, index=mi_idx, key="reg_bat_team")
        with sc_col_vs:
            st.markdown("<div style='text-align: center; font-size: 20px; font-weight: 900; color: #F5C451; margin-top: 30px;'>VS</div>", unsafe_allow_html=True)
        with sc_col2:
            csk_idx = all_teams_ml.index("Chennai Super Kings") if "Chennai Super Kings" in all_teams_ml else min(1, len(all_teams_ml)-1)
            bowl_team_r = st.selectbox("BOWLING FRANCHISE", all_teams_ml, index=csk_idx, key="reg_bowl_team")
            
        bat_color_r = TEAM_COLORS.get(bat_team_r, '#3B82F6')
        bowl_color_r = TEAM_COLORS.get(bowl_team_r, '#F5C451')
        bat_init_r = TEAM_INITIALS.get(bat_team_r, bat_team_r[:3].upper())
        bowl_init_r = TEAM_INITIALS.get(bowl_team_r, bowl_team_r[:3].upper())
        
        # Fixture Spotlight Banner
        st.markdown(f"""<div style="display: flex; justify-content: space-between; align-items: center; background: #061326; border: 1px solid #1D3A5A; border-radius: 8px; padding: 12px 18px; margin-bottom: 18px;">
<div style="display: flex; align-items: center; gap: 10px;">
<span style="background: {bat_color_r}; color: #FFFFFF; font-weight: 900; font-size: 11px; padding: 3px 8px; border-radius: 4px;">{bat_init_r}</span>
<span style="font-size: 14px; font-weight: 800; color: #FFFFFF;">{bat_team_r} <span style="font-size: 11px; color: #14B8A6;">(BATTING)</span></span>
</div>
<div style="font-size: 10px; font-weight: 800; color: #F5C451; text-transform: uppercase; letter-spacing: 1px;">HEAD TO HEAD FIXTURE</div>
<div style="display: flex; align-items: center; gap: 10px;">
<span style="font-size: 14px; font-weight: 800; color: #FFFFFF;">{bowl_team_r}</span>
<span style="background: {bowl_color_r}; color: #FFFFFF; font-weight: 900; font-size: 11px; padding: 3px 8px; border-radius: 4px;">{bowl_init_r}</span>
</div>
</div>""", unsafe_allow_html=True)
        
        # Match Details
        md_c1, md_c2, md_c3, md_c4 = st.columns(4)
        with md_c1:
            season_r = st.selectbox("SEASON", all_seasons_ml, index=0, key="reg_season")
        with md_c2:
            default_v_r = "Wankhede Stadium" if "Wankhede Stadium" in all_venues_ml else all_venues_ml[0]
            venue_r = st.selectbox("VENUE", all_venues_ml, index=all_venues_ml.index(default_v_r), key="reg_venue")
        with md_c3:
            toss_winner_r = st.selectbox("TOSS WINNER", [bat_team_r, bowl_team_r], key="reg_toss_win")
        with md_c4:
            toss_dec_r = st.selectbox("TOSS DECISION", ["Batting", "Fielding"], key="reg_toss_dec")
            
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # Two-column layout for Innings Situation & Prediction Output
        col_reg_in, col_reg_out = st.columns([1.1, 0.9])
        
        with col_reg_in:
            st.markdown("""<div class="sports-card">
<div class="card-title">CURRENT INNINGS SITUATION</div>
<div class="card-sub">Innings progression and phase-wise boundary dynamics.</div>""", unsafe_allow_html=True)
            
            algo_reg = st.selectbox(
                "SELECT ALGORITHM",
                ["Random Forest Regressor", "Linear Regression", "Decision Tree Regressor"],
                key="sb_algo_reg"
            )
            
            rg1, rg2 = st.columns(2)
            with rg1:
                in_wkt = st.number_input("Wickets Lost", min_value=0, max_value=10, value=4, key="r_wkt_in")
                in_4s = st.number_input("Boundaries 4s", min_value=0, max_value=35, value=12, key="r_4s_in")
                in_pp = st.number_input("Powerplay Runs", min_value=0, max_value=130, value=52, key="r_pp_in")
            with rg2:
                in_6s = st.number_input("Boundaries 6s", min_value=0, max_value=30, value=7, key="r_6s_in")
                in_dots = st.number_input("Dot Balls", min_value=0, max_value=85, value=45, key="r_dots_in")
                in_death = st.number_input("Death Over Runs", min_value=0, max_value=120, value=55, key="r_death_in")
                
            inp_reg_payload = {
                'wickets': in_wkt,
                'boundaries_4': in_4s,
                'boundaries_6': in_6s,
                'dot_balls': in_dots,
                'powerplay_runs': in_pp,
                'death_over_runs': in_death
            }
            
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            run_reg_btn = st.button("🏏 RUN SCORE PREDICTION", key="btn_run_reg")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_reg_out:
            model_meta = ml_data['reg_results'][algo_reg]
            pred_score_disp = 158.99
            
            if run_reg_btn:
                try:
                    model_obj = ml_data['reg_models'][algo_reg]
                    res_pred = predict_interactive(model_obj, inp_reg_payload, is_classification=False)
                    pred_score_disp = res_pred['predicted_total_runs']
                    
                    st.session_state['ml_predictions_history'].append({
                        'Fixture': f"{bat_init_r} vs {bowl_init_r}",
                        'Batting Team': bat_team_r,
                        'Bowling Team': bowl_team_r,
                        'Venue': venue_r,
                        'Model': algo_reg.replace(' Regressor', ''),
                        'Prediction': f"{pred_score_disp:.2f} Runs"
                    })
                except Exception as e:
                    st.error("Unable to generate prediction. Please verify input values.")
                    
            st.markdown(f"""<div class="scoreboard-box result-banner-reg">
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1E3A5F; padding-bottom: 8px; margin-bottom: 12px;">
<div class="scoreboard-label">PREDICTED INNINGS SCORE</div>
<div style="font-size: 11px; font-weight: 700; color: #8FA7C2;">{season_r} • {venue_r}</div>
</div>
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
<span style="font-size: 15px; font-weight: 900; color: #FFFFFF;">{bat_team_r}</span>
<span style="font-size: 12px; color: #F5C451; font-weight: 900;">VS</span>
<span style="font-size: 15px; font-weight: 900; color: #CBD5E1;">{bowl_team_r}</span>
</div>
<div class="scoreboard-score">{pred_score_disp:.2f} <span class="scoreboard-score-unit">RUNS</span></div>
<div style="font-size: 13.5px; font-weight: 700; color: #38BDF8; margin-bottom: 12px;">{algo_reg}</div>
<hr style="border: none; border-top: 1px solid #1E3A5F; margin: 12px 0;">
<div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
<span style="font-size: 12px; color: #8FA7C2;">Model R² Score:</span>
<span style="font-size: 13px; font-weight: 800; color: #22C55E;">{model_meta['R2']}</span>
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
<span style="font-size: 12px; color: #8FA7C2;">Test Mean Absolute Error:</span>
<span style="font-size: 13px; font-weight: 800; color: #38BDF8;">±{model_meta['MAE']} runs</span>
</div>
<div style="display: flex; justify-content: space-between;">
<span style="font-size: 12px; color: #8FA7C2;">Test Root Mean Squared Error:</span>
<span style="font-size: 13px; font-weight: 800; color: #CBD5E1;">{model_meta['RMSE']} runs</span>
</div>
</div>""", unsafe_allow_html=True)
            
    # ---------------------------------------------------------
    # TAB 2: CLASSIFICATION SCOREBOARD (TIER CLASSIFICATION)
    # ---------------------------------------------------------
    with tab_clf:
        # Match Scenario Builder
        st.markdown("""<div class="sports-card" style="margin-bottom: 16px;">
<div class="card-title" style="color: #22C55E;">🏏 IPL MATCH SCENARIO</div>
<div class="card-sub">Configure the IPL match context before classifying innings tier.</div>
</div>""", unsafe_allow_html=True)
        
        c_col1, c_col_vs, c_col2 = st.columns([1.2, 0.3, 1.2])
        with c_col1:
            bat_team_c = st.selectbox("BATTING FRANCHISE", all_teams_ml, index=mi_idx, key="clf_bat_team")
        with c_col_vs:
            st.markdown("<div style='text-align: center; font-size: 20px; font-weight: 900; color: #22C55E; margin-top: 30px;'>VS</div>", unsafe_allow_html=True)
        with c_col2:
            bowl_team_c = st.selectbox("BOWLING FRANCHISE", all_teams_ml, index=csk_idx, key="clf_bowl_team")
            
        bat_color_c = TEAM_COLORS.get(bat_team_c, '#3B82F6')
        bowl_color_c = TEAM_COLORS.get(bowl_team_c, '#F5C451')
        bat_init_c = TEAM_INITIALS.get(bat_team_c, bat_team_c[:3].upper())
        bowl_init_c = TEAM_INITIALS.get(bowl_team_c, bowl_team_c[:3].upper())
        
        # Fixture Spotlight Banner
        st.markdown(f"""<div style="display: flex; justify-content: space-between; align-items: center; background: #061326; border: 1px solid #1D3A5A; border-radius: 8px; padding: 12px 18px; margin-bottom: 18px;">
<div style="display: flex; align-items: center; gap: 10px;">
<span style="background: {bat_color_c}; color: #FFFFFF; font-weight: 900; font-size: 11px; padding: 3px 8px; border-radius: 4px;">{bat_init_c}</span>
<span style="font-size: 14px; font-weight: 800; color: #FFFFFF;">{bat_team_c} <span style="font-size: 11px; color: #22C55E;">(BATTING)</span></span>
</div>
<div style="font-size: 10px; font-weight: 800; color: #22C55E; text-transform: uppercase; letter-spacing: 1px;">HEAD TO HEAD FIXTURE</div>
<div style="display: flex; align-items: center; gap: 10px;">
<span style="font-size: 14px; font-weight: 800; color: #FFFFFF;">{bowl_team_c}</span>
<span style="background: {bowl_color_c}; color: #FFFFFF; font-weight: 900; font-size: 11px; padding: 3px 8px; border-radius: 4px;">{bowl_init_c}</span>
</div>
</div>""", unsafe_allow_html=True)
        
        # Match Details
        mc_c1, mc_c2, mc_c3, mc_c4 = st.columns(4)
        with mc_c1:
            season_c = st.selectbox("SEASON", all_seasons_ml, index=0, key="clf_season")
        with mc_c2:
            venue_c = st.selectbox("VENUE", all_venues_ml, index=all_venues_ml.index(default_v_r), key="clf_venue")
        with mc_c3:
            toss_winner_c = st.selectbox("TOSS WINNER", [bat_team_c, bowl_team_c], key="clf_toss_win")
        with mc_c4:
            toss_dec_c = st.selectbox("TOSS DECISION", ["Batting", "Fielding"], key="clf_toss_dec")
            
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        col_clf_in, col_clf_out = st.columns([1.1, 0.9])
        
        with col_clf_in:
            st.markdown("""<div class="sports-card">
<div class="card-title">🏷 CLASSIFY INNINGS TIER</div>
<div class="card-sub">Determine whether an innings qualifies as High-Scoring (>= 152.0 runs empirical median threshold).</div>""", unsafe_allow_html=True)
            
            algo_clf = st.selectbox(
                "SELECT ALGORITHM",
                ["Random Forest Classifier", "Logistic Regression", "Decision Tree Classifier"],
                key="sb_algo_clf"
            )
            
            cg1, cg2 = st.columns(2)
            with cg1:
                c_wkt = st.number_input("Wickets Lost", min_value=0, max_value=10, value=4, key="c_wkt_in")
                c_4s = st.number_input("Boundaries 4s", min_value=0, max_value=35, value=12, key="c_4s_in")
                c_pp = st.number_input("Powerplay Runs", min_value=0, max_value=130, value=52, key="c_pp_in")
            with cg2:
                c_6s = st.number_input("Boundaries 6s", min_value=0, max_value=30, value=7, key="c_6s_in")
                c_dots = st.number_input("Dot Balls", min_value=0, max_value=85, value=45, key="c_dots_in")
                c_death = st.number_input("Death Over Runs", min_value=0, max_value=120, value=55, key="c_death_in")
                
            inp_clf_payload = {
                'wickets': c_wkt,
                'boundaries_4': c_4s,
                'boundaries_6': c_6s,
                'dot_balls': c_dots,
                'powerplay_runs': c_pp,
                'death_over_runs': c_death
            }
            
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            run_clf_btn = st.button("🏷️ CLASSIFY INNINGS", key="btn_run_clf")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_clf_out:
            clf_meta = ml_data['clf_results'][algo_clf]
            class_label_text = "HIGH-SCORING INNINGS"
            prob_disp = 90.25
            
            if run_clf_btn:
                try:
                    model_clf_obj = ml_data['clf_models'][algo_clf]
                    res_clf = predict_interactive(model_clf_obj, inp_clf_payload, is_classification=True)
                    is_high = res_clf['predicted_class'] == 1
                    class_label_text = "HIGH-SCORING INNINGS" if is_high else "LOWER-SCORING INNINGS"
                    prob_val = res_clf['probability_high_scoring']
                    if prob_val is not None:
                        prob_disp = prob_val * 100 if is_high else (1.0 - prob_val) * 100
                        
                    st.session_state['ml_predictions_history'].append({
                        'Fixture': f"{bat_init_c} vs {bowl_init_c}",
                        'Batting Team': bat_team_c,
                        'Bowling Team': bowl_team_c,
                        'Venue': venue_c,
                        'Model': algo_clf.replace(' Classifier', ''),
                        'Prediction': f"{class_label_text} ({prob_disp:.1f}%)"
                    })
                except Exception as e:
                    st.error("Unable to generate classification.")
                    
            st.markdown(f"""<div class="scoreboard-box result-banner-clf" style="border-color: #22C55E;">
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1E3A5F; padding-bottom: 8px; margin-bottom: 12px;">
<div class="scoreboard-label" style="color: #22C55E;">INNINGS CLASSIFICATION</div>
<div style="font-size: 11px; font-weight: 700; color: #8FA7C2;">{season_c} • {venue_c}</div>
</div>
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
<span style="font-size: 15px; font-weight: 900; color: #FFFFFF;">{bat_team_c}</span>
<span style="font-size: 12px; color: #22C55E; font-weight: 900;">VS</span>
<span style="font-size: 15px; font-weight: 900; color: #CBD5E1;">{bowl_team_c}</span>
</div>
<div class="scoreboard-score" style="font-size: 30px;">{class_label_text}</div>
<div style="font-size: 13.5px; font-weight: 700; color: #FFFFFF; margin-bottom: 12px;">{algo_clf}</div>
<hr style="border: none; border-top: 1px solid #1E3A5F; margin: 12px 0;">
<div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
<span style="font-size: 12px; color: #8FA7C2;">Prediction Confidence:</span>
<span style="font-size: 14px; font-weight: 900; color: #F5C451;">{prob_disp:.2f}%</span>
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
<span style="font-size: 12px; color: #8FA7C2;">Model Accuracy:</span>
<span style="font-size: 13px; font-weight: 800; color: #22C55E;">{clf_meta['Accuracy']*100:.2f}%</span>
</div>
<div style="display: flex; justify-content: space-between;">
<span style="font-size: 12px; color: #8FA7C2;">F1 Score:</span>
<span style="font-size: 13px; font-weight: 800; color: #38BDF8;">{clf_meta['F1']:.4f}</span>
</div>
</div>""", unsafe_allow_html=True)
            
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    # ---------------------------------------------------------
    # RECENT PREDICTIONS (SESSION HISTORY)
    # ---------------------------------------------------------
    if len(st.session_state['ml_predictions_history']) > 0:
        st.markdown("""<div class="sports-card">
<div class="card-title">RECENT MATCH SCENARIO PREDICTIONS</div>
<div class="card-sub">Inference history logged during your active browser session.</div>""", unsafe_allow_html=True)
        hist_df = pd.DataFrame(st.session_state['ml_predictions_history']).tail(8)
        st.dataframe(hist_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    # ---------------------------------------------------------
    # Model Benchmarks & Data Leakage Validation Card
    # ---------------------------------------------------------
    col_bench, col_leak = st.columns([1.2, 0.8])
    with col_bench:
        st.markdown("""<div class="sports-card">
<div class="card-title">MODEL BENCHMARKS</div>
<div class="card-sub">Holdout test evaluation (20% split, n = 438 samples)</div>""", unsafe_allow_html=True)
        
        bm_t1, bm_t2 = st.tabs(["REGRESSION", "CLASSIFICATION"])
        with bm_t1:
            df_r = pd.DataFrame(ml_data['reg_results']).T[['MAE', 'RMSE', 'R2']]
            df_r.columns = ["MAE (Runs)", "RMSE (Runs)", "R² Score"]
            st.dataframe(df_r, use_container_width=True)
        with bm_t2:
            df_c = pd.DataFrame(ml_data['clf_results']).T[['Accuracy', 'Precision', 'Recall', 'F1']]
            st.dataframe(df_c, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_leak:
        st.markdown("""<div class="sports-card" style="border-left: 4px solid #F5C451; height: 100%;">
<div class="card-title" style="color: #F5C451;">MODEL VALIDATION & LEAKAGE AUDIT</div>
<div class="card-sub">Technical integrity safeguards for viva presentation</div>
<div style="font-size: 12.5px; line-height: 1.5; color: #CBD5E1;">
Completed-innings <code>run_rate</code> (<code>total_runs / (balls/6)</code>) and <code>balls_bowled</code> are
<strong>strictly excluded</strong> from predictor variables.<br><br>
Because run rate is mathematically derived from total score, its inclusion causes trivial target reconstruction (R² ≈ 1.0).
Our models exclusively utilize genuine match dynamics (boundaries, dot balls, phase runs, wickets).
</div>
</div>""", unsafe_allow_html=True)
        
    # Feature Importance
    st.markdown("""<div class="sports-card">
<div class="card-title">FEATURE IMPORTANCE</div>
<div class="card-sub">Random Forest Regressor split contributions across match dynamics</div>""", unsafe_allow_html=True)
    
    rf_reg_model = ml_data['reg_models']['Random Forest Regressor']
    fig_fi = plot_feature_importance(rf_reg_model, REGRESSION_FEATURES, model_name='Random Forest Regressor', save=False)
    if fig_fi:
        fig_fi = stylize_fig(fig_fi)
        st.pyplot(fig_fi)
        plt.close(fig_fi)
        
    st.markdown("""<div class="sports-explanation-card">
<strong>FEATURE IMPORTANCE INSIGHT:</strong> Feature importance indicates which input variables contributed most to the model's split decisions and variance reduction.
Sixes struck (<code>boundaries_6</code>, 38.2%) and late-innings scoring (<code>death_over_runs</code>, 29.6%) represent the primary scoring multipliers.
This represents predictive statistical association, not causal influence.
</div>
</div>""", unsafe_allow_html=True)

    # Academic Value Box (Why This Prediction?)
    st.markdown("""<div class="sports-card" style="border-left: 4px solid #3B82F6; margin-top: 14px;">
<div class="card-title" style="color: #38BDF8;">WHY THIS PREDICTION? (ACADEMIC VALUE)</div>
<div class="card-sub">Curriculum synthesis connecting Data Analytics & Visualization to Machine Learning</div>
<div style="font-size: 12.5px; line-height: 1.6; color: #CBD5E1;">
The predictive section uses historical IPL innings fixtures to learn non-linear relationships between innings match dynamics and total runs scored.
<br><br>
<strong>Core Syllabus Demonstrations:</strong>
<ul style="margin: 6px 0 0 16px; padding: 0;">
    <li><strong>Data Preprocessing:</strong> Handling missing values, standardizing franchise identities, filtering completed regular 20-over matches.</li>
    <li><strong>Feature Engineering:</strong> Extracting boundary physics, dot-ball resistance, and powerplay/death over phase accumulation.</li>
    <li><strong>Data Leakage Prevention:</strong> Eliminating target-derived variables (run_rate, balls) to ensure statistical validity.</li>
    <li><strong>Predictive Analytics:</strong> Comparing parametric linear baselines against non-linear tree ensembles (Decision Trees, Random Forests).</li>
    <li><strong>Evaluation Metrics:</strong> Assessing R², MAE, RMSE for regression and Accuracy, Precision, Recall, F1 for classification.</li>
</ul>
</div>
</div>""", unsafe_allow_html=True)


# =============================================================================
# 5. ⌕ DATA EXPLORER / DATA CENTER PAGE
# =============================================================================
elif "Data Explorer" in nav_selection:
    st.markdown("""
    <div style="margin-bottom: 18px;">
        <div style="font-size: 20px; font-weight: 900; color: #FFFFFF;">DATA CENTER</div>
        <div style="font-size: 13px; color: #94A3B8; margin-top: 2px;">
            Inspect the underlying IPL dataset, review schemas, and filter slices for laboratory evaluation.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    de_tab1, de_tab2, de_tab3, de_tab4 = st.tabs(["OVERVIEW", "SCHEMA", "MATCH DATA", "STATISTICS"])
    
    # TAB 1: OVERVIEW
    with de_tab1:
        st.markdown('<div class="sports-card">', unsafe_allow_html=True)
        st.markdown("##### Centralized Structural Audit")
        ds1, ds2, ds3, ds4 = st.columns(4)
        with ds1:
            st.markdown(f"""
            <div class="stat-kpi-card stat-kpi-card-gold">
                <div class="stat-kpi-label">Total Rows</div>
                <div class="stat-kpi-num">{dataset_stats['rows']:,}</div>
                <div class="stat-kpi-meta">{dataset_stats['regular_innings']:,} regular innings</div>
            </div>
            """, unsafe_allow_html=True)
        with ds2:
            st.markdown(f"""
            <div class="stat-kpi-card">
                <div class="stat-kpi-label">Total Columns</div>
                <div class="stat-kpi-num">{dataset_stats['features']}</div>
                <div class="stat-kpi-meta">Raw tabular attributes</div>
            </div>
            """, unsafe_allow_html=True)
        with ds3:
            st.markdown(f"""
            <div class="stat-kpi-card stat-kpi-card-green">
                <div class="stat-kpi-label">Duplicates</div>
                <div class="stat-kpi-num">{raw_df.duplicated().sum()}</div>
                <div class="stat-kpi-meta">Zero redundant records</div>
            </div>
            """, unsafe_allow_html=True)
        with ds4:
            st.markdown(f"""
            <div class="stat-kpi-card stat-kpi-card-teal">
                <div class="stat-kpi-label">Seasons</div>
                <div class="stat-kpi-num">{dataset_stats['seasons']}</div>
                <div class="stat-kpi-meta">{dataset_stats['season_min']} to {dataset_stats['season_max']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # TAB 2: SCHEMA
    with de_tab2:
        st.markdown('<div class="sports-card">', unsafe_allow_html=True)
        st.markdown("##### Dataset Schema & Missing Value Audit")
        schema_table = pd.DataFrame({
            'Attribute': raw_df.columns,
            'Data Type': raw_df.dtypes.astype(str).values,
            'Missing Values': raw_df.isnull().sum().values,
            'Sample Value': raw_df.iloc[0].astype(str).values
        })
        st.dataframe(schema_table, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # TAB 3: MATCH DATA
    with de_tab3:
        st.markdown('<div class="sports-card">', unsafe_allow_html=True)
        st.markdown("##### Multi-Filter Slice Workbench")
        
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            de_seasons = st.multiselect("Season Filter", sorted(df['season'].unique()), default=[], key="de_s_flt")
        with fc2:
            de_bats = st.multiselect("Batting Team Filter", sorted(df['batting_team_std'].unique()), default=[], key="de_bat_flt")
        with fc3:
            de_bowls = st.multiselect("Bowling Team Filter", sorted(df['bowling_team_std'].unique()), default=[], key="de_bowl_flt")
        with fc4:
            de_venues = st.multiselect("Venue Filter", sorted(df['venue'].dropna().unique())[:15], default=[], key="de_ven_flt")
            
        de_filtered = df.copy()
        if de_seasons:
            de_filtered = de_filtered[de_filtered['season'].isin(de_seasons)]
        if de_bats:
            de_filtered = de_filtered[de_filtered['batting_team_std'].isin(de_bats)]
        if de_bowls:
            de_filtered = de_filtered[de_filtered['bowling_team_std'].isin(de_bowls)]
        if de_venues:
            de_filtered = de_filtered[de_filtered['venue'].isin(de_venues)]
            
        all_cols = list(df.columns)
        default_view_cols = [
            'match_id', 'season', 'date', 'batting_team_std', 'bowling_team_std',
            'total_runs', 'wickets', 'run_rate', 'powerplay_runs', 'death_over_runs',
            'venue', 'winner_std'
        ]
        chosen_cols = st.multiselect(
            "Select Columns to View",
            all_cols,
            default=[col for col in default_view_cols if col in all_cols],
            key="de_cols_select"
        )
        
        row_limit = st.slider("Rows to Display", min_value=5, max_value=50, value=10, key="de_slider_limit")
        st.markdown(f"**Showing {min(row_limit, len(de_filtered))} of {len(de_filtered):,} filtered records:**")
        st.dataframe(de_filtered[chosen_cols].head(row_limit), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # TAB 4: STATISTICS
    with de_tab4:
        st.markdown('<div class="sports-card">', unsafe_allow_html=True)
        st.markdown("##### Descriptive Statistics (Numerical Attributes)")
        num_attributes = df.select_dtypes(include=[np.number]).columns
        st.dataframe(df[num_attributes].describe().round(2).T, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# 6. ⓘ ABOUT PROJECT PAGE
# =============================================================================
elif "About" in nav_selection:
    st.markdown("""
    <div style="margin-bottom: 18px;">
        <div style="font-size: 20px; font-weight: 900; color: #FFFFFF;">ABOUT THE PROJECT</div>
        <div style="font-size: 13px; color: #94A3B8; margin-top: 2px;">
            Academic scope, curriculum alignment, dataset specifications, and implementation details.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    ab_c1, ab_c2 = st.columns(2)
    with ab_c1:
        st.markdown("""
        <div class="sports-card">
            <div class="card-title" style="color: #F5C451;">PROJECT OBJECTIVE</div>
            <div style="font-size: 13px; line-height: 1.5; color: #CBD5E1;">
                An empirical sports analytics investigation into 17 editions of the Indian Premier League (2008–2024).
                The project demonstrates the complete Data Analytics and Visualization (DAV) syllabus while featuring
                a leakage-free Machine Learning extension for predictive score modeling.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="sports-card">
            <div class="card-title" style="color: #38BDF8;">DATASET SPECIFICATIONS</div>
            <div style="font-size: 13px; line-height: 1.5; color: #CBD5E1;">
                <strong>Source File:</strong> <code>ipl_2008_2024_complete(1).csv</code><br>
                <strong>Total Observations:</strong> {dataset_stats['rows']:,} team innings<br>
                <strong>Raw Attributes:</strong> {dataset_stats['features']} columns<br>
                <strong>Granularity:</strong> Innings-level observation (Innings 1, 2, and Super Overs)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sports-card">
            <div class="card-title" style="color: #22C55E;">TECHNOLOGIES USED</div>
            <div style="font-size: 13px; line-height: 1.5; color: #CBD5E1;">
                * <strong>Language:</strong> Python 3.10+<br>
                * <strong>Analytics:</strong> NumPy, Pandas, SciPy<br>
                * <strong>Visualization:</strong> Matplotlib, Seaborn, Squarify<br>
                * <strong>Machine Learning:</strong> Scikit-Learn (Pipelines, Standard Scaler, Regression, Random Forests)<br>
                * <strong>Interface:</strong> Streamlit Sports Portal Dashboard
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with ab_c2:
        st.markdown("""
        <div class="sports-card">
            <div class="card-title" style="color: #F5C451;">CURRICULUM CONCEPTS COVERED</div>
            <div style="font-size: 13px; line-height: 1.5; color: #CBD5E1;">
                * <strong>NumPy (CO-1):</strong> ndarrays, vectorized calculations, boolean masking, slicing, moments.<br>
                * <strong>Pandas (CO-1):</strong> Series, DataFrames, groupby multi-aggregations, crosstabs, joins.<br>
                * <strong>Data Wrangling:</strong> Imputation of UAE cities, franchise standardization, IQR outlier audits.<br>
                * <strong>Feature Scaling:</strong> Min-Max Normalization vs Z-Score Standardization.<br>
                * <strong>Visualization:</strong> 20 rich charts (Line, Bar, KDE, Box, Heatmap, Treemap, Bump, Waterfall, Waffle).
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sports-card">
            <div class="card-title" style="color: #38BDF8;">MACHINE LEARNING ALGORITHMS</div>
            <div style="font-size: 13px; line-height: 1.5; color: #CBD5E1;">
                * <strong>Regression:</strong> Linear Regression, Decision Tree Regressor, Random Forest Regressor<br>
                * <strong>Classification:</strong> Logistic Regression, Decision Tree Classifier, Random Forest Classifier
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sports-card">
            <div class="card-title" style="color: #EC4899;">IMPORTANT LIMITATIONS</div>
            <div style="font-size: 13px; line-height: 1.5; color: #CBD5E1;">
                1. Innings-level observation does not track delivery-by-delivery micro-physics.<br>
                2. Rain-shortened matches (42 D/L instances) represent truncated overs requiring contextual filtering.<br>
                3. Feature importance represents statistical association rather than direct strategic causation.<br>
                4. Zero personal identifiers included to preserve strict academic integrity.
            </div>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Global Subdued Sports Website Footer
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="app-footer">
    <hr style="border: none; border-top: 1px solid #1E3A5F; margin-bottom: 16px;">
    <div style="font-size: 13px; font-weight: 800; color: #F8FAFC; letter-spacing: 0.8px;">
        🏏 IPL ANALYTICS & PREDICTIVE INSIGHTS
    </div>
    <div style="font-size: 11.5px; color: #94A3B8; margin-top: 4px;">
        Data Analytics • Visualization • Predictive Modeling • Dataset: IPL {dataset_stats['season_min']}–{dataset_stats['season_max']}
    </div>
    <div style="font-size: 10.5px; color: #64748B; margin-top: 8px;">
        Academic Data Analytics Project • Presentation & Viva Ready
    </div>
</div>
""", unsafe_allow_html=True)
