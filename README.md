## What this notebook does

This notebook builds a machine learning model that predicts the **Load_Type** of an electrical grid — either `Light_Load`, `Medium_Load`, or `Maximum_Load` — using power consumption and time-based data collected every 15 minutes through 2018.

The dataset (`load_data.csv`) has 35,041 rows and 9 columns. Missing values were present across almost every numeric column, so the first step was cleaning and preparing the data before feeding it to any model.

## Steps I followed

**1. Data loading and exploration**

Loaded the CSV, checked shape, data types, missing value counts, and duplicates. No duplicate rows were found, but several columns had missing values — `Usage_kWh` had the most (1,559), followed by `Leading_Current_Power_Factor` (1,471). A heatmap was plotted to visualise where the gaps were.

**2. Cleaning and feature engineering**

- Converted `Date_Time` from string to datetime using the `%d-%m-%Y %H:%M` format.
- Sorted the dataframe chronologically.
- Extracted `Hour`, `Minute`, `Day`, `Month`, `DayOfWeek`, `IsWeekend`, and `WeekOfYear` from the timestamp. These help the model pick up daily and weekly usage patterns.

**3. Train / test split**

Since this is time-series data, I used a **chronological split** instead of a random one — this avoids data leakage. Training set is Jan–Nov 2018 (32,064 rows) and test set is December 2018 (2,977 rows).

**4. Preprocessing**

- Missing values filled with the **median** (`SimpleImputer`). Fit on training data only, then applied to test — this is important so the test set doesn't influence the imputation.
- Target column `Load_Type` encoded with `LabelEncoder`, again fit only on training data.

**5. Model training**

Trained four classifiers to compare performance:

| Model | Notes |
|---|---|
| Logistic Regression | Wrapped in a pipeline with `StandardScaler` |
| Decision Tree | Default depth |
| Random Forest | 200 trees |
| Extra Trees | 200 trees |

All models use `random_state=42` so results are reproducible.

**6. Evaluation**

Evaluated each model on the December test set using accuracy, precision, recall, and F1 score (all weighted for the multiclass case).

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Random Forest** | **0.9207** | 0.9300 | 0.9207 | **0.9223** |
| Decision Tree | 0.8962 | 0.9132 | 0.8962 | 0.8983 |
| Extra Trees | 0.8801 | 0.8948 | 0.8801 | 0.8827 |
| Logistic Regression | 0.6446 | 0.7040 | 0.6446 | 0.6637 |

Random Forest came out on top, so it was chosen as the final model.

## Results

**Per-class performance (Random Forest):**
- Light_Load — precision 0.99, recall 0.93, F1 0.96
- Maximum_Load — precision 0.92, recall 0.84, F1 0.87
- Medium_Load — precision 0.79, recall 0.97, F1 0.87

The confusion matrix shows `Maximum_Load` is the hardest class — the model sometimes confuses it with `Medium_Load`, which makes sense since the two overlap in usage patterns.

**Top features from Random Forest:**
1. `Lagging_Current_Reactive.Power_kVarh`
2. `Usage_kWh`
3. `Lagging_Current_Power_Factor`
4. `NSM`

## Visualisations included

- Missing values heatmap
- Load type distribution (countplot)
- Load type distribution by month
- Model comparison bar chart
- Confusion matrix
- Feature importance bar plot

## Manual prediction

The last two cells let you test the model on a hand-crafted input. I entered sample values (Usage 4.5 kWh, Hour 8, etc.) and the model predicted `Light_Load` with 90% confidence. The cell also prints probabilities for each class so you can see how sure the model is.

## Dependencies

pandas, numpy, matplotlib, seaborn, scikit-learn. Just run `pip install` on those if anything's missing and run the notebook top to bottom.

## Files

- `BharathMV_Assignment2.ipynb` — the notebook
- `load_data.csv` — dataset 
-`Readme.txt` - instructions

## Note

Random Forest's performance could probably be pushed higher with a bit of class balancing (since `Light_Load` makes up over half the data) or some hyperparameter tuning. I left it as is because the baseline result already looks strong, but that's a natural next step if accuracy matters more than simplicity.
