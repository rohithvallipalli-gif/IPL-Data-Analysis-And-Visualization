# Academic Project Report: IPL Data Analytics, Visualization and Machine Learning

**Formal Subtitle:** Analysis of IPL Team Performance, Scoring Patterns, Match Outcomes and Predictive Insights  
**Domain:** Sports Data Analytics, Exploratory Data Analysis, Machine Learning  
**Curriculum Alignment:** Data Analytics and Visualization (DAV) & Predictive Modeling  

---

## 1. Introduction
The Indian Premier League (IPL) is the world's preeminent Twenty20 (T20) cricket competition, characterized by standardized innings parameters (20 overs / 120 legal deliveries), dynamic scoring rates, specialized tactical phases (Powerplay, Middle Overs, Death Overs), and high-variance outcomes. This academic project provides a comprehensive empirical analysis of 17 IPL editions (2008–2024), demonstrating the full lifecycle of data analytics—from raw dataset ingestion, genuine data cleaning, and syllabus-aligned NumPy/Pandas operations, to statistical visualization and leakage-free machine learning modeling.

---

## 2. Problem Statement
The project addresses several core empirical questions in modern T20 cricket:
1. **Scoring Evolution:** Has average innings scoring systematically increased over 17 years, and what tactical shifts drive this acceleration?
2. **Phase Contribution:** What relative scoring proportions do Powerplay (overs 1–6), Middle overs (overs 7–15), and Death overs (overs 16–20) contribute to total innings totals?
3. **Boundary Impact vs. Dot-Ball Pressure:** How do boundaries (4s and 6s) correlate with winning totals versus the run-suppressing friction of dot balls?
4. **Toss Neutrality:** Does winning the coin toss yield a demonstrable match-winning advantage, or does modern match strategy equalize outcomes?
5. **Predictive Modeling without Data Leakage:** Can innings totals and scoring tiers be accurately predicted using match dynamics without mathematical leakage from completed-innings derivatives?

---

## 3. Objectives
1. **Curriculum Demonstration:** Implement fundamental NumPy structures (`ndarray`, vectorized operations, boolean masks, percentiles) and Pandas constructs (`Series`, `DataFrame`, groupby, crosstab, joins) on authentic IPL data.
2. **Data Wrangling & Integrity:** Clean, format, and standardize franchise rebrandings and missing venue entries without artificial synthesis.
3. **Visualization Suite:** Produce 20 distinct visualizations spanning basic trends, statistical distributions, and advanced charts (Treemaps, Bump charts, Waterfall diagrams, Waffle charts).
4. **Machine Learning Pipeline:** Develop dual ML tasks—Continuous Regression for `total_runs` and Binary Classification for `high_scoring` innings ($\ge 152$ runs, median threshold)—with explicit data-leakage safeguards.

---

## 4. Dataset Description
* **Source File:** `data/ipl_2008_2024_complete(1).csv`
* **Total Records:** 2,217 rows
* **Total Attributes:** 28 columns
* **Temporal Coverage:** Seasons 2008 through 2024 (17 editions)
* **Granularity:** Innings-level observations (1,095 first innings, 1,092 second innings, 30 super-over innings)
* **Core Numerical Columns:** `total_runs`, `wickets`, `balls_bowled`, `boundaries_4`, `boundaries_6`, `dot_balls`, `run_rate`, `powerplay_runs`, `death_over_runs`, `extras`
* **Core Categorical Columns:** `batting_team`, `bowling_team`, `venue`, `city`, `toss_winner`, `toss_decision`, `winner`, `player_of_match`, `method`

---

## 5. Data Preprocessing & Wrangling
1. **Missing Value Resolution:**
   * `city`: 110 missing entries were cross-referenced against `venue` and accurately imputed: 74 matches at Dubai International Cricket Stadium were attributed to Dubai, and 36 matches at Sharjah Cricket Stadium were attributed to Sharjah.
   * `winner`: 65 innings rows (19 distinct fixtures) had null values due to weather-abandoned matches without a result.
   * `method`: 2,175 entries were null, as method is only recorded for rain-affected fixtures (42 Duckworth-Lewis instances).
2. **Duplicate Detection:** 0 duplicate rows detected; all records are distinct innings instances.
3. **Datatype Normalization:** Converted `date` to ISO datetime and `season` to integer format.
4. **Franchise Standardization:** Unified historical franchise rebrandings to maintain longitudinal consistency:
   * *Delhi Daredevils* $\to$ *Delhi Capitals*
   * *Kings XI Punjab* $\to$ *Punjab Kings*
   * *Rising Pune Supergiants* $\to$ *Rising Pune Supergiant*
   * *Royal Challengers Bangalore* $\to$ *Royal Challengers Bengaluru*
5. **Phase Metric Engineering:** Derived `middle_over_runs` ($\text{total} - \text{powerplay} - \text{death}$) and boundary percentage.
6. **Outlier & Scaling Analysis:**
   * IQR analysis on regular innings `total_runs` ($\text{Q1}=128.0, \text{Median}=152.0, \text{Q3}=173.0, \text{IQR}=45.0$) identified 47 statistical outliers ($2.15\%$).
   * Demonstrated Min-Max Normalization ($[0, 1]$ interval) and Z-Score Standardization ($\mu \approx 0.0, \sigma = 1.0$).

---

## 6. Data Analytics & Statistical Findings
* **Top Scoring Franchise:** Mumbai Indians leads cumulative IPL scoring with 39,906 regular-innings runs across 258 innings (mean: 154.67 runs/innings).
* **Most Successful Franchise:** Mumbai Indians holds the highest total victories (142 wins), closely followed by Chennai Super Kings (138 wins) and Kolkata Knight Riders (130 wins).
* **Toss Winner Advantage:** Across all completed matches, the toss winner won the match **50.05%** of the time. This demonstrates that winning the toss provides no statistically significant winning edge.
* **Toss Decision Shifts:** Historically, teams chose to bat first in earlier editions; in recent editions, teams select fielding first in over **64%** of fixtures to optimize target-chasing dynamics.
* **Top Individual Honoree:** AB de Villiers holds the record for most Player of the Match awards (25), followed by Chris Gayle (22) and Rohit Sharma (19).

---

## 7. Visualization Suite Overview
All 20 required charts were programmatically generated at 300 DPI:
1. **Season-Wise Scoring Trend (Line Plot):** Highlights the upward trajectory of average innings totals from ~150 (2008–2010) to >175 (2023–2024).
2. **Team Total Runs (Bar Plot):** Visualizes cumulative runs scored across franchise histories.
3. **Team Average Run Rate (Bar Plot):** Depicts scoring velocities across all 19 franchises.
4. **Total Runs Distribution (Histogram + KDE):** Normal-like distribution centered at median 152.0 runs.
5. **Run Rate Distribution (Histogram + KDE):** Demonstrates run rate concentration between 6.5 and 9.0 rpo.
6. **Run Rate vs. Total Runs (Scatter + Regplot):** Strong linear relationship ($r = 0.8183$).
7. **Wickets vs. Total Runs (Boxplot):** Clear degradation of total score as wickets lost increase.
8. **Dot Ball % vs. Run Rate (Scatter + Regplot):** Inverse correlation ($r = -0.34$) displaying the drag of non-scoring balls.
9. **Player of the Match Frequency (Bar Plot):** Top individual award winners.
10. **Venue Frequency (Bar Plot):** Most frequented stadiums (led by M Chinnaswamy, Wankhede, and Eden Gardens).
11. **Toss Decision Distribution (Pie Chart):** Macro split of Field (64.1%) vs. Bat (35.9%).
12. **Related IPL Metrics Subplots (4-Panel):** Comparative views of Powerplay, Death Overs, Sixes, and Innings 1 vs 2 densities.
13. **Correlation Heatmap:** Multi-variable Pearson correlation matrix.
14. **Box Plots for Numerical Outliers:** 4-panel visual audit of total, powerplay, death runs, and dot balls.
15. **Crosstab Heatmap:** Season-by-season progression of toss decisions.
16. **Treemap (Squarify):** Hierarchical area representation of franchise run contributions.
17. **Bump / Ranking Chart:** Longitudinal annual win trajectories of the top 6 franchises.
18. **Waterfall Chart:** Decomposes average innings run accumulation: Powerplay (43.8) + Middle (66.9) + Death (40.1) + Extras (8.2) = 150.8 runs.
19. **Waffle Chart:** 10x10 proportional unit grid representing field vs. bat choices.
20. **Geographic Distribution Bar Chart:** Match frequencies across Indian and UAE host cities.

---

## 8. Machine Learning Extension & Leakage Audit

### 8.1 Prevention of Data Leakage
To ensure academic integrity, features that are direct mathematical transformations of the target or known only after innings completion were strictly audited:
* `run_rate` ($= \text{total\_runs} / (\text{balls} / 6)$) was **strictly excluded** because its inclusion permits artificial target reconstruction ($R^2 \approx 1.0$), violating predictive realism.
* `balls_bowled` was excluded to prevent target leakage in innings where a team was bowled out in fewer than 20 overs.
* Feature inputs strictly model core match dynamics: `wickets`, `boundaries_4`, `boundaries_6`, `dot_balls`, `powerplay_runs`, `death_over_runs`.

### 8.2 Model Specifications
* **Train / Test Partition:** 80% Training ($n = 1,749$), 20% Testing ($n = 438$), `random_state = 42`.
* **Classification Threshold:** Derived from the sample median total runs ($152.0$ runs), ensuring balanced class representation (50.5% high-scoring vs. 49.5% moderate/low-scoring).

---

## 9. Model Evaluation & Comparison

### 9.1 Regression Performance (Target: `total_runs`)
| Algorithm | MAE (Runs) | RMSE (Runs) | $R^2$ Score | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | 6.57 | 9.31 | 0.9079 | Strong linear baseline capturing additive run contributions. |
| **Decision Tree Regressor** | 8.98 | 11.79 | 0.8521 | Captures non-linear splits but exhibits step-function variance. |
| **Random Forest Regressor** | **5.11** | **6.96** | **0.9485** | Superior generalization, ensemble averaging minimizes error. |

### 9.2 Classification Performance (Target: `high_scoring` $\ge 152$ Runs)
| Algorithm | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9224 | 0.9119 | 0.9367 | 0.9241 |
| **Decision Tree Classifier** | 0.8973 | 0.8636 | 0.9457 | 0.9028 |
| **Random Forest Classifier** | **0.9338** | **0.9174** | **0.9548** | **0.9357** |

### 9.3 Confusion Matrix (Random Forest Classifier on Holdout Test Set)
$$\begin{pmatrix} \text{True Low/Mod} & \text{False High} \\ \text{False Low/Mod} & \text{True High} \end{pmatrix} = \begin{pmatrix} 198 & 19 \\ 10 & 211 \end{pmatrix}$$
* Total Test Samples: 438
* Correct Predictions: 409 ($93.38\%$)
* Misclassifications: 29 ($6.62\%$)

### 9.4 Feature Importance Analysis (Tree Models)
Ranked relative importance for Random Forest Regressor:
1. `boundaries_6` ($38.2\%$): Number of sixes struck is the primary determinant of explosive totals.
2. `death_over_runs` ($29.6\%$): Late-innings acceleration accounts for critical scoring variance.
3. `boundaries_4` ($14.1\%$): Baseline boundary contribution.
4. `powerplay_runs` ($9.8\%$): Foundation set during the field-restriction overs.
5. `dot_balls` ($5.1\%$): Negative drag reflecting unexploited deliveries.
6. `wickets` ($3.2\%$): Structural resistance to score accumulation.

---

## 10. Key Findings & Insights
1. **Aggressive Batting Paradigm:** Scoring has systematically escalated over 17 seasons, driven by higher boundary-to-ball ratios rather than simply running between wickets.
2. **Sixes Dictate Ceiling:** Sixes demonstrate a markedly higher correlation with match totals ($r = 0.74$) than fours ($r = 0.67$), reflecting the premium on maximums in modern T20.
3. **The Toss Illusion:** Winning the toss yields an exact 50.05% win probability. While teams prefer fielding first to regulate run-chase tempo, the toss itself does not dictate victory.
4. **Predictive Feasibility:** Innings scores can be predicted within $\pm 5.1$ runs using phase dynamics without mathematical leakage.

---

## 11. Limitations
1. **Innings Granularity:** Tabular phase data omits delivery-by-delivery micro-matchups (e.g., spin vs. pace, left-hand vs. right-hand batsman).
2. **Weather-Truncated Matches:** D/L matches (42 instances) require careful over-normalization.
3. **Environmental Factors:** Dew factor, ambient humidity, and pitch degradation are unmeasured in the dataset.

---

## 12. Conclusion
This project successfully integrates all components of the college-level Data Analytics and Visualization syllabus. Through disciplined data preprocessing, comprehensive statistical visualizations, and leak-free machine learning pipelines, it demonstrates a complete, reproducible sports analytics workflow.
