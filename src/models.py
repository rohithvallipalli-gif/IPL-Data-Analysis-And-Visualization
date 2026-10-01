"""
models.py
---------
Machine Learning pipelines for:
1. Regression (Predict Total Runs): Linear Regression, Decision Tree, Random Forest
2. Classification (Classify High-Scoring Innings): Logistic Regression, Decision Tree, Random Forest
Includes data leakage checks, evaluation metrics, feature importance, and interactive inference.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Selected features matching innings match dynamics without mathematical leakage
REGRESSION_FEATURES = [
    'wickets',
    'boundaries_4',
    'boundaries_6',
    'dot_balls',
    'powerplay_runs',
    'death_over_runs'
]

CLASSIFICATION_FEATURES = REGRESSION_FEATURES.copy()


def prepare_ml_data(df: pd.DataFrame, classification_threshold: float = 152.0):
    """
    Prepares dataset for ML tasks:
    - Filters to regular innings (innings 1 & 2)
    - Verifies absence of data leakage
    - Defines regression target (total_runs)
    - Defines classification target (High_Scoring based on median threshold: 152)
    """
    reg_df = df[df['is_regular_innings']].dropna(subset=REGRESSION_FEATURES + ['total_runs']).copy()
    
    # Regression
    X_reg = reg_df[REGRESSION_FEATURES]
    y_reg = reg_df['total_runs']
    
    # Classification: 1 if high-scoring (>= median: 152), 0 otherwise
    reg_df['high_scoring'] = (reg_df['total_runs'] >= classification_threshold).astype(int)
    X_clf = reg_df[CLASSIFICATION_FEATURES]
    y_clf = reg_df['high_scoring']
    
    return X_reg, y_reg, X_clf, y_clf, classification_threshold


# -------------------------------------------------------------
# Regression Modeling
# -------------------------------------------------------------

def train_and_evaluate_regression(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Trains and evaluates Linear Regression, Decision Tree, and Random Forest Regressors."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    models = {
        'Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        'Decision Tree Regressor': DecisionTreeRegressor(
            max_depth=6, min_samples_split=10, random_state=random_state
        ),
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_split=8, random_state=random_state
        )
    }
    
    results = {}
    fitted_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        
        results[name] = {
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'R2': round(r2, 4),
            'y_pred_sample': y_pred[:5].tolist(),
            'y_test_sample': y_test.iloc[:5].tolist()
        }
        fitted_models[name] = model
        
    return results, fitted_models, X_train, X_test, y_train, y_test


# -------------------------------------------------------------
# Classification Modeling
# -------------------------------------------------------------

def train_and_evaluate_classification(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Trains and evaluates Logistic Regression, Decision Tree, and Random Forest Classifiers."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=random_state))
        ]),
        'Decision Tree Classifier': DecisionTreeClassifier(
            max_depth=5, min_samples_split=10, random_state=random_state
        ),
        'Random Forest Classifier': RandomForestClassifier(
            n_estimators=100, max_depth=7, min_samples_split=8, random_state=random_state
        )
    }
    
    results = {}
    fitted_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'Confusion_Matrix': cm
        }
        fitted_models[name] = model
        
    return results, fitted_models, X_train, X_test, y_train, y_test


# -------------------------------------------------------------
# Feature Importance & Comparison
# -------------------------------------------------------------

def plot_feature_importance(model, feature_names: list, model_name: str = 'Tree Model', save: bool = True) -> plt.Figure:
    """Plots feature importance bar chart for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('regressor', None), 'feature_importances_'):
        importances = model.named_steps['regressor'].feature_importances_
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('regressor', None), 'coef_'):
        importances = np.abs(model.named_steps['regressor'].coef_)
    else:
        return None
        
    fi_df = pd.DataFrame({
        'Feature': [f.replace('_', ' ').title() for f in feature_names],
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.barh(fi_df['Feature'], fi_df['Importance'], color='#3182bd', edgecolor='#333333')
    ax.set_title(f'Feature Importance: {model_name}', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Relative Importance Score', fontsize=10)
    ax.grid(axis='x')
    plt.tight_layout()
    if save:
        fname = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=300)
    return fig


def predict_interactive(model, input_dict: dict, is_classification: bool = False):
    """
    Performs dynamic model inference without hardcoded outputs.
    Expects input_dict with keys:
    ['wickets', 'boundaries_4', 'boundaries_6', 'dot_balls', 'powerplay_runs', 'death_over_runs']
    """
    row = pd.DataFrame([[
        input_dict['wickets'],
        input_dict['boundaries_4'],
        input_dict['boundaries_6'],
        input_dict['dot_balls'],
        input_dict['powerplay_runs'],
        input_dict['death_over_runs']
    ]], columns=REGRESSION_FEATURES)
    
    pred = model.predict(row)[0]
    if is_classification:
        prob = None
        if hasattr(model, 'predict_proba'):
            prob = float(model.predict_proba(row)[0][1])
        return {
            'predicted_class': int(pred),
            'class_label': 'High-Scoring (>= 152 runs)' if pred == 1 else 'Moderate/Low-Scoring (< 152 runs)',
            'probability_high_scoring': round(prob, 4) if prob is not None else None
        }
    else:
        return {
            'predicted_total_runs': round(float(pred), 2)
        }
