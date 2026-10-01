"""
test_project.py
---------------
End-to-end integration test for the IPL Data Analytics, Visualization,
and Machine Learning project. Tests real data loading, wrangling,
NumPy/Pandas operations, all 20 visualizations, and all 6 ML models.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure UTF-8 stdout on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure src/ is importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.data_processing import (
    load_raw_data,
    inspect_data,
    clean_and_wrangle_data,
    detect_outliers_iqr,
    normalize_min_max,
    standardize_z_score
)
from src.analysis import (
    demonstrate_numpy_operations,
    demonstrate_pandas_operations,
    answer_all_ipl_questions
)
from src.visualization import generate_all_visualizations
from src.models import (
    prepare_ml_data,
    train_and_evaluate_regression,
    train_and_evaluate_classification,
    plot_feature_importance,
    predict_interactive,
    REGRESSION_FEATURES
)


def run_all_tests():
    print("==================================================")
    print("STARTING IPL PROJECT INTEGRATION TESTS")
    print("==================================================")
    
    # 1. Dataset Loading
    csv_path = os.path.join(BASE_DIR, 'data', 'ipl_2008_2024_complete(1).csv')
    print(f"\n[1] Testing Dataset Loading from: {csv_path}")
    raw_df = load_raw_data(csv_path)
    assert not raw_df.empty, "Loaded dataframe is empty!"
    print(f" -> Raw dataset shape: {raw_df.shape} (Expected: 2217 rows, 28 columns)")
    
    # 2. Dataset Inspection
    print("\n[2] Testing Dataset Inspection")
    inspection = inspect_data(raw_df)
    assert inspection['shape'] == (2217, 28), f"Unexpected shape {inspection['shape']}"
    assert inspection['duplicate_count'] == 0, f"Found duplicates: {inspection['duplicate_count']}"
    print(f" -> Columns count: {len(inspection['columns'])}")
    print(f" -> Missing values in city: {inspection['missing_values'].get('city', 0)}")
    print(f" -> Missing values in winner: {inspection['missing_values'].get('winner', 0)}")
    print(f" -> Innings distribution: {inspection['innings_counts']}")
    
    # 3. Data Wrangling
    print("\n[3] Testing Data Cleaning and Wrangling")
    df_clean = clean_and_wrangle_data(raw_df, standardize_teams=True)
    assert pd.api.types.is_datetime64_any_dtype(df_clean['date']), "Date column is not datetime64!"
    assert df_clean['season'].dtype in ['int32', 'int64'], "Season is not integer!"
    assert df_clean['city'].isna().sum() == 0, f"Unresolved city NaNs: {df_clean['city'].isna().sum()}"
    print(f" -> City NaNs resolved via venue mapping! Clean shape: {df_clean.shape}")
    print(f" -> Regular 20-over innings count: {df_clean['is_regular_innings'].sum()}")
    
    # Outlier, Normalization, Standardization
    outlier_res = detect_outliers_iqr(df_clean[df_clean['is_regular_innings']]['total_runs'])
    print(f" -> Total Runs IQR Outliers: {outlier_res['outlier_count']} ({outlier_res['outlier_percentage']}%)")
    norm_runs = normalize_min_max(df_clean['total_runs'])
    assert round(norm_runs.min(), 2) == 0.0 and round(norm_runs.max(), 2) == 1.0, "Normalization bounds error"
    std_runs = standardize_z_score(df_clean['total_runs'])
    assert abs(std_runs.mean()) < 1e-4, "Standardization mean error"
    print(" -> Normalization & Standardization verified.")
    
    # 4. NumPy Operations
    print("\n[4] Testing NumPy Syllabus Concepts")
    numpy_res = demonstrate_numpy_operations(df_clean)
    assert numpy_res['ndarray_info']['type'] == 'ndarray', "NumPy ndarray test failed"
    print(f" -> NumPy ndarray shape: {numpy_res['ndarray_info']['shape']}, dtype: {numpy_res['ndarray_info']['dtype']}")
    print(f" -> Innings >= 200 runs (boolean mask): {numpy_res['boolean_masking']['count_above_200']}")
    print(f" -> Mean calculated run rate: {numpy_res['vectorized_metrics']['mean_calculated_run_rate']}")
    
    # 5. Pandas Operations
    print("\n[5] Testing Pandas Syllabus Concepts")
    pandas_res = demonstrate_pandas_operations(df_clean)
    assert 'season_agg' in pandas_res, "Pandas season aggregation failed"
    print(f" -> Season aggregation shape: {pandas_res['season_agg'].shape}")
    print(" -> Crosstab sample:\n", pandas_res['toss_crosstab'])
    
    # 6. IPL Analytics Questions (1-15)
    print("\n[6] Testing Answers to 15 IPL Analytics Questions")
    answers = answer_all_ipl_questions(df_clean)
    print(f" -> Top run scoring team: {answers['team_total_runs'].iloc[0]['batting_team_std']} ({answers['team_total_runs'].iloc[0]['total_runs']:,} runs)")
    print(f" -> Top winning team: {answers['team_wins'].iloc[0]['team']} ({answers['team_wins'].iloc[0]['wins']} wins)")
    print(f" -> Run rate & Total runs correlation: {answers['rr_runs_corr']}")
    print(f" -> Toss winner won match: {answers['toss_win_pct']}% of the time")
    print(f" -> Top Player of the Match: {answers['top_potm'].iloc[0]['player']} ({answers['top_potm'].iloc[0]['awards_count']} awards)")
    
    # 7. Visualization Generation (all 20 charts)
    print("\n[7] Testing Generation of all 20 Required Visualizations")
    figs = generate_all_visualizations(df_clean)
    assert len(figs) == 20, f"Expected 20 figures, got {len(figs)}"
    figures_dir = os.path.join(BASE_DIR, 'outputs', 'figures')
    saved_images = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
    print(f" -> Successfully generated and verified {len(saved_images)} figures in {figures_dir}")
    assert len(saved_images) >= 20, "Not all figures were saved!"
    
    # 8. Machine Learning Pipeline & Evaluation
    print("\n[8] Testing Machine Learning Pipelines")
    X_reg, y_reg, X_clf, y_clf, threshold = prepare_ml_data(df_clean)
    print(f" -> ML Features: {REGRESSION_FEATURES}")
    print(f" -> Classification Threshold: {threshold} runs (Data-driven median)")
    
    # 8.1 Regression
    print("\n--- Regression Models ---")
    reg_results, reg_models, _, _, _, _ = train_and_evaluate_regression(X_reg, y_reg)
    for model_name, metrics in reg_results.items():
        print(f"Model: {model_name} -> MAE: {metrics['MAE']}, RMSE: {metrics['RMSE']}, R²: {metrics['R2']}")
        assert metrics['R2'] > 0.70, f"{model_name} R² score unexpectedly low!"
        
    # 8.2 Classification
    print("\n--- Classification Models ---")
    clf_results, clf_models, _, _, _, _ = train_and_evaluate_classification(X_clf, y_clf)
    for model_name, metrics in clf_results.items():
        print(f"Model: {model_name} -> Accuracy: {metrics['Accuracy']}, Precision: {metrics['Precision']}, Recall: {metrics['Recall']}, F1: {metrics['F1']}")
        assert metrics['Accuracy'] > 0.75, f"{model_name} Accuracy unexpectedly low!"
        
    # 8.3 Feature Importance
    print("\n--- Feature Importance ---")
    rf_reg = reg_models['Random Forest Regressor']
    fi_fig = plot_feature_importance(rf_reg, REGRESSION_FEATURES, model_name='Random Forest Regressor')
    assert fi_fig is not None, "Feature importance plot failed"
    print(" -> Feature importance chart created successfully.")
    
    # 8.4 Dynamic Inference Test (User specified input test)
    print("\n--- Dynamic Interactive Prediction Test ---")
    sample_input = {
        'wickets': 4,
        'boundaries_4': 12,
        'boundaries_6': 7,
        'dot_balls': 45,
        'powerplay_runs': 52,
        'death_over_runs': 55
    }
    pred_res = predict_interactive(rf_reg, sample_input, is_classification=False)
    print(f"Input: {sample_input}")
    print(f"Dynamic Predicted Total Runs: {pred_res['predicted_total_runs']}")
    assert 140 <= pred_res['predicted_total_runs'] <= 195, "Prediction out of realistic T20 bounds"
    
    rf_clf = clf_models['Random Forest Classifier']
    pred_clf = predict_interactive(rf_clf, sample_input, is_classification=True)
    print(f"Dynamic Classification Result: {pred_clf['class_label']} (Probability: {pred_clf['probability_high_scoring']})")
    
    print("\n==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY! SOURCE OF TRUTH VERIFIED.")
    print("==================================================")


if __name__ == '__main__':
    run_all_tests()
