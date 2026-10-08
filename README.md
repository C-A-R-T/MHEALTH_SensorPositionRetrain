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

The gap between the leading classifiers (below 0.012) is much smaller than the spread between people (about 0.05). A Wilcoxon test between Extra Trees and XGBoost gives p = 0.945. **Which classifier is chosen matters less than who wears the sensors.** The largest errors come from two people whose behaviour differs from the rest (see `report/table_diagnosis.csv`):
- Subject 3 sat with the arm hanging, so sitting is predicted as standing.
- Subject 6 jogged as hard as the others ran.

### RQ3: Masking overstates feature importance
Removing the 10 features ranked highest by each method (score with all features: 0.947):

| Method | Masked, not retrained | Retrained (ROAR) |
|---|---|---|
| TreeSHAP | 0.370 | 0.932 |
| KernelSHAP | 0.445 | 0.912 |
| LIME | 0.312 | 0.933 |
| Permutation | 0.536 | 0.907 |
| Gain | 0.373 | 0.948 |
| Random removal | 0.942 | 0.950 |

- Masking drops the score to 0.3–0.5, while retraining loses at most 0.04. The reason: the sensors carry overlapping information, so the retrained model recovers it from the remaining features.
- ROAR **cannot tell** which explanation method is more faithful.

### RQ3: The chest device can be dropped
| Setting (LightGBM, leave-one-subject-out) | Macro-F1 | Worst person |
|---|---|---|
| All 3 positions | 0.934 | 0.867 |
| **Drop the chest device (acceleration + ECG)** | **0.894** | 0.809 |
| Drop the wrist | 0.865 | 0.748 |
| Drop the ankle | 0.851 | 0.742 |
| Ankle only / chest only / wrist only | 0.792 / 0.786 / 0.749 | 0.53–0.64 |
| ECG only | 0.118 | 0.053 |

- The explanations give the ankle about 40% of the attribution, the wrist about 36%, the chest about 23% and the ECG about 1%. This order **matches the drop-one-position experiment**.
- **Practical rule:** wear the ankle and wrist sensors and drop the chest device, at a cost of about 0.04 macro-F1. A single sensor is not enough, because for some people only about half of the activities are then recognised correctly.

## 5. Limitations

- Only 10 people, so differences below about 0.03 are not reliable enough to draw conclusions.
- ROAR uses one split (people 8–10) and one model (LightGBM). LightGBM is used because TreeSHAP and Gain need a boosted tree model.
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
2. **`Notebook_B_MHEALTH_SensorPositionRetrain_executed.ipynb`** (about 22 min on an 8-thread CPU): checks the md5 against the manifest, runs the 10 classifiers and the RQ3 experiments, and draws the SHAP figure.

On a correct run, the md5 of `data/processed/mhealth.parquet` is `3d81b90d136d90450e7650e80612e8e9`.

## 7. Repository layout

```
├── Notebook_A_MHEALTH_SensorPositionRetrain.ipynb            data, cleaning, SQL, RQ1
├── Notebook_B_MHEALTH_SensorPositionRetrain_executed.ipynb   classifiers (RQ2), explanations (RQ3)
├── sql/queries.sql                  queries Q1–Q5 (DuckDB)
├── data/processed/manifest.json     rows, columns and md5 of the handover file
├── report/
│   ├── table_q*.csv, table_rq1_tests.csv          RQ1
│   ├── table_rq2.csv, table_rq2_folds.csv         RQ2 (mean and per person)
│   ├── table_rq3_roar.csv                         masking vs retraining
│   ├── table_rq3_positions.csv                    one position only and drop one position
│   ├── table_rq3_group_share.csv, table_rq3_methods.csv, table_top10_by_method.csv
│   ├── table_diagnosis.csv                        causes of the largest errors, with evidence
│   └── fig_*.png                                  figures
└── requirements.txt                 library versions that ran both notebooks
```

## 8. Key references

- Banos, O., et al. (2014). mHealthDroid: A novel framework for agile development of mobile health applications. *IWAAL 2014*, LNCS 8868, 91–98. https://doi.org/10.1007/978-3-319-13105-4_14
- Hooker, S., Erhan, D., Kindermans, P.-J., & Kim, B. (2019). A benchmark for interpretability methods in deep neural networks. *NeurIPS 32*.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *KDD 2016*. https://doi.org/10.1145/2939672.2939778
