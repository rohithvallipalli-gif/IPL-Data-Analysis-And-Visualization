# IPL Data Analytics, Visualization and Machine Learning

### Analysis of IPL Team Performance, Scoring Patterns, Match Outcomes and Predictive Insights

An end-to-end academic data science and sports analytics project exploring 17 editions of the Indian Premier League (2008–2024). The project is built with an **75% Data Analytics & Visualization (DAV)** focus and a **25% Machine Learning** extension.

---

## 1. Problem Statement
In Twenty20 (T20) cricket, match outcomes and scoring dynamics are often judged through subjective commentary. This project applies empirical statistical analysis and predictive modeling to answer:
1. How has T20 scoring velocity escalated across the 17 IPL seasons?
2. What are the relative scoring contributions of the Powerplay, Middle, and Death overs?
3. How do boundary frequency and dot-ball suppression dictate match totals?
4. Does winning the coin toss grant an empirical advantage in winning the match?
5. Can final innings scores and high-scoring matches be reliably predicted without data leakage?

---

## 2. Objectives
* **Course Syllabus Coverage:** Complete implementation of **NumPy** (`ndarray`, boolean masking, slicing, vectorization, percentiles) and **Pandas** (`Series`, `DataFrame`, groupby aggregations, crosstab, relational joins).
* **Genuine Data Wrangling:** Date parsing, franchise rebrand standardization, missing venue/city imputation, outlier detection (IQR), and feature scaling (Min-Max vs. Z-Score).
* **20 Rich Visualizations:** Comprehensive chart suite spanning Matplotlib and Seaborn (Line, Bar, Scatter, Box, KDE Histograms, Correlation Heatmap, Subplots, Treemap, Bump chart, Waterfall chart, and Waffle chart).
* **Leakage-Free Machine Learning:** Dual predictive tasks (Regression for continuous runs, Classification for high-scoring innings) comparing 6 algorithms with dynamic inference and feature importance.

---

## 3. Dataset Specifications
* **Dataset File:** `data/ipl_2008_2024_complete(1).csv`
* **Dimensions:** 2,217 rows $\times$ 28 columns
* **Seasons:** 2008 through 2024 (17 completed editions)
* **Granularity:** Innings-level observations (1,095 first innings, 1,092 second innings, 30 super overs)
* **Key Attributes:**
  * Performance: `total_runs`, `wickets`, `balls_bowled`, `boundaries_4`, `boundaries_6`, `dot_balls`, `run_rate`
  * Phases: `powerplay_runs`, `powerplay_wickets`, `death_over_runs`, `death_over_wickets`, `extras`
  * Match Context: `season`, `date`, `venue`, `city`, `toss_winner`, `toss_decision`, `winner`, `player_of_match`, `method`

---

## 4. Technologies
* **Language:** Python 3.10+ (tested on Python 3.13)
* **Numerical & Data Processing:** NumPy, Pandas, SciPy
* **Visualization:** Matplotlib, Seaborn, Squarify
* **Machine Learning:** Scikit-Learn (Pipelines, Linear Models, Tree Regressors/Classifiers, Ensemble Random Forests)
* **Interactive Application:** Streamlit
* **Notebook Development:** Jupyter, nbformat, nbclient, ipykernel

---

## 5. Data Preprocessing & Wrangling
* **City Imputation:** 110 missing city entries were resolved by cross-referencing venues (Dubai International Cricket Stadium $\to$ Dubai; Sharjah Cricket Stadium $\to$ Sharjah).
* **Winner Missing Values:** 65 innings rows (19 distinct fixtures) were documented as abandoned matches without result.
* **Franchise Name Standardization:** Unified historical name transitions (*Delhi Daredevils* $\to$ *Delhi Capitals*; *Kings XI Punjab* $\to$ *Punjab Kings*; *Rising Pune Supergiants* $\to$ *Rising Pune Supergiant*; *Royal Challengers Bangalore* $\to$ *Royal Challengers Bengaluru*).
* **Phase Decomposition:** Engineered `middle_over_runs` ($\text{total} - \text{powerplay} - \text{death}$) and boundary percentage.
* **Outlier Analysis:** Interquartile Range (IQR) analysis on `total_runs` ($\text{Q1}=128.0, \text{Median}=152.0, \text{Q3}=173.0, \text{IQR}=45.0$) identified 47 outliers ($2.15\%$).
* **Feature Scaling:** Implemented Min-Max Normalization ($[0, 1]$) and Z-Score Standardization ($\mu \approx 0.0, \sigma = 1.0$).

---

## 6. IPL Analytics Summary
1. **Scoring Trajectory:** Average innings scores rose from ~150 runs in 2008–2010 to >175 runs in 2023–2024.
2. **Top Run Scoring Franchise:** Mumbai Indians (39,906 runs across 258 regular innings).
3. **Most Successful Franchise:** Mumbai Indians (142 wins), followed by Chennai Super Kings (138 wins) and Kolkata Knight Riders (130 wins).
4. **Toss Neutrality:** Across all completed matches, the toss winner won the match **50.05%** of the time, proving the toss grants no inherent victory advantage.
5. **Toss Decision Preference:** In recent seasons, franchises select fielding first in **>64%** of matches to control chase targets.
6. **Top Player of the Match:** AB de Villiers (25 awards), Chris Gayle (22 awards), Rohit Sharma (19 awards).

---

## 7. Required Visualizations (20 Charts)
All figures are saved at 300 DPI in `outputs/figures/`:
1. `01_season_scoring_trend.png` — Season-wise average score line plot
2. `02_team_total_runs.png` — Franchise cumulative runs bar plot
3. `03_team_avg_run_rate.png` — Historical team run rate bar plot
4. `04_total_runs_distribution.png` — Total runs histogram with KDE and median/mean markers
5. `05_run_rate_distribution.png` — Run rate histogram with KDE
6. `06_run_rate_vs_total_runs.png` — Scatter plot with linear regression fit ($r = 0.8183$)
7. `07_wickets_vs_total_runs.png` — Boxplot of innings total runs across wickets lost
8. `08_dot_ball_pct_vs_run_rate.png` — Scatter plot demonstrating dot ball drag ($r = -0.34$)
9. `09_player_of_match_frequency.png` — Bar plot of top 12 award recipients
10. `10_venue_frequency.png` — Top 10 most hosted IPL venues
11. `11_toss_decision_distribution.png` — Pie chart of Field (64.1%) vs Bat (35.9%)
12. `12_related_metrics_subplots.png` — 4-panel subplots dashboard
13. `13_correlation_heatmap.png` — Multi-variable Pearson correlation matrix
14. `14_numerical_boxplots.png` — 4-panel boxplots for numerical outlier audit
15. `15_crosstab_visualization.png` — Heatmap of toss decision proportion across 17 seasons
16. `16_treemap_team_runs.png` — Squarify treemap of franchise run contributions
17. `17_bump_ranking_chart.png` — Annual win trajectories of major franchises
18. `18_waterfall_innings_breakdown.png` — Waterfall buildup of average innings phases
19. `19_waffle_chart_toss.png` — 100-cell unit waffle grid of toss decisions
20. `20_geographic_city_matches.png` — Distribution of fixtures across Indian and UAE host cities

---

## 8. Machine Learning Extension

### Data Leakage Safeguard
Features mathematically derived from completed innings totals (such as `run_rate = total_runs / (balls/6)`) or end-of-innings delivery limits (`balls_bowled`) are strictly excluded to avoid artificial target reconstruction. Models strictly utilize legitimate match dynamics: `wickets`, `boundaries_4`, `boundaries_6`, `dot_balls`, `powerplay_runs`, `death_over_runs`.

### Regression Performance (Target: `total_runs`)
| Algorithm | MAE (Runs) | RMSE (Runs) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 6.57 | 9.31 | 0.9079 |
| **Decision Tree Regressor** | 8.98 | 11.79 | 0.8521 |
| **Random Forest Regressor** | **5.11** | **6.96** | **0.9485** |

### Classification Performance (Target: `high_scoring` $\ge 152$ Runs)
| Algorithm | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9224 | 0.9119 | 0.9367 | 0.9241 |
| **Decision Tree Classifier** | 0.8973 | 0.8636 | 0.9457 | 0.9028 |
| **Random Forest Classifier** | **0.9338** | **0.9174** | **0.9548** | **0.9357** |

---

## 9. Project Structure
```text
IPL-Data-Analytics/
│
├── data/
│   ├── ipl_2008_2024_complete(1).csv
│   └── ipl_2008_2024_complete.csv
│
├── notebooks/
│   └── IPL_Data_Analytics_and_Visualization.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── analysis.py
│   ├── visualization.py
│   └── models.py
│
├── outputs/
│   ├── figures/
│   │   ├── 01_season_scoring_trend.png
│   │   ├── ...
│   │   └── 20_geographic_city_matches.png
│   └── IPL_Project_Report.md
│
├── app.py
├── test_project.py
├── test_streamlit_app.py
├── build_notebook.py
├── requirements.txt
└── README.md
```

---

## 10. Installation & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run End-to-End Tests
```bash
# Core pipeline and visualization test
python test_project.py

# Automated Streamlit UI verification test
python test_streamlit_app.py
```

### 3. Build & Execute Master Jupyter Notebook
```bash
python build_notebook.py
```
Or launch directly with Jupyter:
```bash
jupyter notebook notebooks/IPL_Data_Analytics_and_Visualization.ipynb
```

### 4. Launch Interactive Application
```bash
streamlit run app.py
```
The application provides six structured sections in the sidebar:
* `🏠 Overview`: Macro dataset summary metrics (2,217 rows, 28 features, 17 seasons, 19 franchises) and introductory scoring trend.
* `📊 IPL Analytics`: Interactive season and team filters updating team totals, phase metrics, toss decisions, and venues.
* `📈 Visualizations`: Interactive explorer for all 20 charts with explanatory academic cards.
* `🤖 Machine Learning`: Dual tabs for score estimation (Regression) and high-scoring categorization (Classification) with live metrics and feature importances.
* `🔍 Data Explorer`: Interactive Pandas table inspection, column multiselect, and filtering.
* `ℹ️ About Project`: Academic summary of objectives, dataset specifications, algorithms, and limitations.

---

## 11. Limitations
1. **Innings-Level Granularity:** Does not track delivery-by-delivery matchup physics (e.g. ball speed, pitch trajectory).
2. **Weather-Shortened Matches:** Rain-affected D/L matches (42 instances) require careful over-normalization.
3. **No Tactical Causation:** Feature importance reflects predictive association rather than direct strategic causation.
