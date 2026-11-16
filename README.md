# Choosing Body Positions for Wearable Activity Recognition by Retraining-validated Explanations on MHEALTH with Ten Classifiers

Assignment 2, ADY201m course.

---

## 1. Problem

Mobile health apps often ask users to **wear several sensors** (chest, wrist, ankle, …) so that they can recognise whether the person is walking, running, sitting or lying. Every extra device makes people more likely to stop wearing them, so designers want to know: **which body position can be dropped without losing much accuracy?**

A common approach is to read an explanation plot (SHAP, LIME, …) and drop the sensor ranked "least important". This has two problems:

1. **Explanations are rarely verified.** The usual check masks a feature (replaces it with its mean) and measures how much the score falls. The model then receives inputs unlike anything it was trained on, so the score collapses and importance is **overstated**.
2. **Many MHEALTH studies split windows at random**, so the same person appears in both the training and the test set and scores are close to perfect. In practice a model always has to work on **a new person it has never seen**.

This study checks explanations by **retraining** (ROAR, *Remove And Retrain*) and evaluates every model with **person-wise splits**.

## 2. Data

[MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) (Banos et al., 2014), UCI Machine Learning Repository, CC BY 4.0 licence.

- 10 people, 12 activities: standing, sitting, lying, walking, climbing stairs, waist bend, arm raise, knee bend, cycling, jogging, running, jumping.
- 50 Hz sensors at 3 positions: **chest** (acceleration + 2 ECG leads), **left ankle** and **right wrist** (acceleration, gyroscope, magnetometer).
- 1,215,745 raw rows, about 28% of them labelled with an activity.
- The signals are cut into 2 s windows with a 0.2 s step, and the mean and standard deviation of each channel are computed per window. This gives **33,148 windows** and **42 features**. The magnetometer is left out because it depends on the room.

The dataset is **not included in this repository**; Notebook A downloads it from UCI.

## 3. Research questions

| | Question | Approach |
|---|---|---|
| **RQ1** | Which body position moves in which activity, and how much do people differ? | SQL (DuckDB), coefficient of variation between people, Friedman test |
| **RQ2** | Which of 10 classifiers from 4 families recognises activities best on **an unseen person**? | Leave-one-subject-out: each of the 10 people is held out in turn as the test set. Macro-F1, a naive baseline and a Wilcoxon test. |
| **RQ3** | Are the features that 5 explanation methods rank as important still important after **retraining**? Is the position favoured by the explanations the best position to keep? | ROAR with TreeSHAP, KernelSHAP, LIME, Permutation and Gain; retraining with one position only or with one position removed; rank stability between two groups of people |

**10 classifiers:** Logistic Regression, Linear SVM, k-NN, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting, LightGBM, XGBoost, small MLP.

## 4. Main results

### RQ1: Each position "sees" a different group of activities
| Activity | Chest | Ankle | Wrist |
|---|---|---|---|
| standing / sitting / lying | 0.14–0.16 | 0.10 | 0.10–0.12 |
| arm raise | 0.31 | **0.11** | **2.75** |
| walking | 2.20 | **5.20** | 2.21 |
| running | 8.58 | 7.01 | 6.70 |

*(Standard deviation of acceleration magnitude, m/s²; larger means more movement.)*

- The ankle separates locomotion activities, and the wrist separates arm movements.
- People differ most in free-form movements: waist bend has a CV of 33%, arm raise 27%.

### RQ2: The top five classifiers are equivalent
| Classifier | Macro-F1 (mean ± sd over 10 people) |
|---|---|
| Extra Trees | 0.962 ± 0.048 |
| XGBoost | 0.958 ± 0.050 |
| Logistic Regression | 0.955 ± 0.043 |
| HistGB | 0.951 ± 0.049 |
| Random Forest | 0.950 ± 0.056 |
| LightGBM | 0.933 ± 0.053 |
| Naive baseline (most frequent class) | 0.014 |

The gap between the leading classifiers (below 0.012) is much smaller than the spread between people (about 0.05). A Wilcoxon test between Extra Trees and XGBoost gives p = 0.945, and an equivalence test (TOST, margin ±0.03) finds most pairs among the top five equivalent (`report/table_rq2_tests.csv`). **Which classifier is chosen matters less than who wears the sensors.** The largest errors come from two people whose behaviour differs from the rest (see `report/table_diagnosis.csv`):
- Subject 3 sat with the arm hanging, so sitting is predicted as standing.
- Subject 6 jogged as hard as the others ran.

### RQ3: Masking overstates feature importance
Removing the 10 features ranked highest by each method (people 8–10 held out, LightGBM, mean over 3 seeds; score with all features: 0.947):

| Method | Masked, not retrained | Retrained (ROAR) |
|---|---|---|
| TreeSHAP | 0.368 | 0.930 |
| KernelSHAP | 0.464 | 0.923 |
| LIME | 0.368 | 0.956 |
| Permutation | 0.500 | 0.905 |
| Gain | 0.382 | 0.944 |
| Random removal | 0.937 | 0.951 |

- Masking drops the score to 0.3–0.5, while retraining loses at most about 0.04. The reason: the sensors carry overlapping information, so the retrained model recovers it from the remaining features.
- The same gap holds on **5 different splits of held-out people and for both LightGBM and Extra Trees**: masked 0.24–0.62 vs retrained 0.88–0.92 (mean over splits); the gap exceeds 0.1 on 5/5 splits for every method except Extra Trees + Permutation (4/5). See `report/table_rq3_roar_splits.csv` and `report/table_rq3_roar_summary.csv`.
- ROAR **cannot tell** which explanation method is more faithful.

### RQ3: The chest device costs the least to drop
Leave-one-subject-out, mean over 10 people and 3 seeds, for three tree models (`report/table_rq3_positions.csv`, figure `report/fig_rq3_positions_3models.png`):

| Setting | LightGBM | Extra Trees | XGBoost |
|---|---|---|---|
| All 3 positions | 0.937 | 0.957 | 0.958 |
| **Drop the chest device (acceleration + ECG)** | **0.896** | **0.932** | **0.900** |
| Drop the wrist | 0.867 | 0.885 | 0.893 |
| Drop the ankle | 0.853 | 0.878 | 0.857 |
| Ankle only / chest only / wrist only | 0.79 / 0.79 / 0.74 | 0.80 / 0.79 / 0.75 | 0.80 / 0.79 / 0.72 |
| ECG only | 0.120 | 0.118 | 0.122 |

- For all three models, dropping the chest device costs the least (0.026–0.058 macro-F1), and the order chest < wrist ≈ ankle is the same.
- The cost is **not negligible**: the TOST test does not show equivalence within ±0.03, and the loss is significant for LightGBM and XGBoost (`report/table_rq3_positions_tests.csv`).
- The explanations give the ankle about 40% of the attribution, the wrist about 36%, the chest about 23% and the ECG about 1%. A model-free baseline (mutual information) and per-feature attribution point the same way (`report/table_rq3_selection_baselines.csv`).
- **Practical rule (on MHEALTH):** if one position has to go, drop the chest device; it is the smallest loss, not a free one. A single sensor is not enough, because for some people only about half of the activities are then recognised correctly.

### External check on PAMAP2 (`Notebook_C_PAMAP2_ExternalCheck_executed.ipynb`)
PAMAP2 (Reiss and Stricker, 2012, UCI, CC BY 4.0) has the same three positions: dominant wrist, chest and dominant ankle, plus a heart-rate strap on the chest. Same feature recipe (mean and SD of ±16 g acceleration, gyroscope and acceleration magnitude per position, plus heart rate; 44 features), 2 s windows with a 1 s step (19,178 windows), 8 people (subject 109 performed almost none of the protocol), 12 activities. Not every person did every activity, so macro-F1 is averaged over the activities each held-out person performed.

Cost of dropping one position (macro-F1 lost, leave-one-subject-out, mean over people and 3 seeds; `report/table_external_check.csv`):

| Data set | Model | Drop chest device | Drop wrist | Drop ankle |
|---|---|---|---|---|
| MHEALTH | LightGBM | **0.042** | 0.070 | 0.084 |
| MHEALTH | Extra Trees | **0.026** | 0.072 | 0.079 |
| MHEALTH | XGBoost | **0.058** | 0.065 | 0.101 |
| PAMAP2 | LightGBM | **0.047** | 0.093 | 0.056 |
| PAMAP2 | Extra Trees | **0.028** | 0.083 | 0.046 |
| PAMAP2 | XGBoost | **0.026** | 0.080 | 0.043 |

- On both data sets and for all three models, the chest device is the cheapest position to drop. With all positions, PAMAP2 reaches 0.845–0.880 macro-F1 (figure `report/fig_pamap2_positions_3models.png`).
- The order of the other two positions differs: on PAMAP2 the wrist matters most, probably because ironing, vacuum cleaning and nordic walking involve the arms.
- **The explanations and mutual information do not pick the right position on PAMAP2.** Both give the wrist the lowest score (TreeSHAP gives the chest device, chest IMU + heart rate, the largest share; mutual information ranks the ankle first) and would drop the wrist first, yet the wrist is the most expensive position to lose. Only retraining shows that the chest device is the cheapest to drop (`report/pamap2_selection_baselines.csv`).
- Masking again overstates importance: removing the top 10 features drops the score to 0.35–0.59 when masked and to 0.74–0.84 when retrained (`report/pamap2_roar.csv`; people 106–108 held out, LightGBM and Extra Trees).
- With 8 people the cost of dropping the chest device is significant only for Extra Trees and unresolved for LightGBM and XGBoost; dropping the wrist is significant for all three (`report/pamap2_positions_tests.csv`).

## 5. Limitations

- Two data sets with few people (10 and 8), so differences below about 0.03 are not reliable enough to draw conclusions. ROAR is repeated on 5 splits and 2 models, but the per-person spread is still large compared with the differences between positions.
- The features are simple statistics, and there is no thigh sensor, so sitting and standing are easily confused.
- The ECG feature is the standard deviation of the voltage, **not heart rate**.
- Model hyperparameters are fixed and have not been tuned.

## 6. How to reproduce

Python 3.11 is required. On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Open the notebooks in VS Code (Jupyter extension) or Jupyter, and run them in this order:

1. **`Notebook_A_MHEALTH_SensorPositionRetrain.ipynb`** (about 15 s, plus downloading the 72 MB dataset): downloads the data, cuts windows, cleans, runs the SQL, draws the RQ1 figure, and writes `data/processed/mhealth.parquet` and `manifest.json`.
2. **`Notebook_B_MHEALTH_SensorPositionRetrain_executed.ipynb`** checks the md5 against the manifest, runs the 10 classifiers and the RQ3 experiments, and draws the SHAP figure. The revised RQ3 experiments (5 ROAR splits, 3 models × 3 seeds for positions) take a few hours; every step saves its results to `report/*.csv` and resumes from them, and RQ2 is loaded from `report/table_rq2_folds.csv` unless `RERUN_RQ2 = True`.

3. **`Notebook_C_PAMAP2_ExternalCheck_executed.ipynb`** (optional, about 1.5 h on 2 CPU cores): downloads PAMAP2 from UCI (about 690 MB), cuts windows and repeats the position, ROAR, mutual-information and TOST experiments. Results are saved to `report/pamap2_*.csv` and resumed from there.

4. **`Notebook_D_Paper_Figures_executed.ipynb`** (about 1 min, no training): draws every figure and table of the paper and the slides from the CSV files in `report/`, so the paper always shows the numbers in this repository.

On a correct run, the md5 of `data/processed/mhealth.parquet` is `3d81b90d136d90450e7650e80612e8e9`.

## 7. Repository layout

```
├── Notebook_A_MHEALTH_SensorPositionRetrain.ipynb            data, cleaning, SQL, RQ1
├── Notebook_B_MHEALTH_SensorPositionRetrain_executed.ipynb   classifiers (RQ2), explanations (RQ3)
├── Notebook_C_PAMAP2_ExternalCheck_executed.ipynb            external check on PAMAP2
├── Notebook_D_Paper_Figures_executed.ipynb                   paper and slide figures from report/*.csv
├── sql/queries.sql                  queries Q1–Q5 (DuckDB)
├── data/processed/manifest.json     rows, columns and md5 of the handover file
├── report/
│   ├── table_q*.csv, table_rq1_tests.csv          RQ1
│   ├── table_rq2.csv, table_rq2_folds.csv         RQ2 (mean and per person)
│   ├── table_rq2_tests.csv                        RQ2 pairwise tests (Wilcoxon, TOST ±0.03)
│   ├── table_rq3_roar.csv                         masking vs retraining (people 8–10)
│   ├── table_rq3_roar_splits.csv, _summary.csv    ROAR on 5 splits × 2 models
│   ├── table_rq3_positions.csv, _folds.csv        one position only / drop one, 3 models × 3 seeds
│   ├── table_rq3_positions_tests.csv              position tests (TOST ±0.03)
│   ├── table_rq3_selection_baselines.csv          mutual information vs attribution vs retrain cost
│   ├── table_rq3_group_share.csv, table_rq3_methods.csv, table_top10_by_method.csv
│   ├── pamap2_*.csv, fig_pamap2_positions_3models.png   external check on PAMAP2
│   ├── table_external_check.csv                   MHEALTH vs PAMAP2: cost of dropping each position
│   ├── paper/                                     Fig. 1-6 and Tables I-III (tables_paper.tex) of the paper
│   ├── slides/                                    the same results with large fonts for the talk
│   ├── table_diagnosis.csv                        causes of the largest errors, with evidence
│   └── fig_*.png                                  figures
└── requirements.txt                 library versions that ran both notebooks
```

## 8. Key references

- Banos, O., et al. (2014). mHealthDroid: A novel framework for agile development of mobile health applications. *IWAAL 2014*, LNCS 8868, 91–98. https://doi.org/10.1007/978-3-319-13105-4_14
- Reiss, A., & Stricker, D. (2012). Introducing a new benchmarked dataset for activity monitoring. *ISWC 2012*, 108–109. https://doi.org/10.1109/ISWC.2012.13
- Hooker, S., Erhan, D., Kindermans, P.-J., & Kim, B. (2019). A benchmark for interpretability methods in deep neural networks. *NeurIPS 32*.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *KDD 2016*. https://doi.org/10.1145/2939672.2939778
