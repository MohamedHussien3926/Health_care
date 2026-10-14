# Healthcare Predictive Analytics: Diabetic Patient Readmission

A machine learning project that predicts whether a diabetic patient will be **readmitted to the hospital within 30 days** of discharge. The goal is to help healthcare professionals identify high-risk patients early and use hospital resources better.

## Dataset

**Diabetes 130-US Hospitals (1999–2008)**, from the UCI Machine Learning Repository.

- 101,766 hospital encounters, 50 columns
- Patient demographics, diagnoses, lab results, medications, and hospital stay information
- Target: `readmitted` (`NO`, `>30`, `<30`)
- `IDS_mapping.csv` explains the meaning of the ID columns (`admission_type_id`, `discharge_disposition_id`, `admission_source_id`)

## Project Roadmap

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Data collection, exploration & preprocessing | Done |
| 2 | Data analysis, visualization & feature engineering | Planned |
| 3 | Model development & optimization | Planned |
| 4 | MLOps, deployment & monitoring | Planned |
| 5 | Final documentation & presentation | Planned |

## Milestone 1: What was done

**Exploration**
- Checked missing values (written as `?`), data types, and distributions
- Found outliers using boxplots and histograms
- Found that the target is imbalanced: only about 11% of encounters are readmitted within 30 days

**Preprocessing**
- Dropped `weight` (97% missing), `payer_code`, and constant columns (`examide`, `citoglipton`)
- Removed rows with invalid gender and patients who died or went to hospice
- Kept only the first encounter of each patient to avoid data leakage
- Filled missing categories with `Unknown` / `Not tested`
- Converted ID columns to readable groups using the mapping file
- Grouped ICD-9 diagnosis codes into 9 clinical categories
- Converted age ranges to numbers
- Capped outliers at the 99th percentile
- Created `n_drugs_used` and `n_drugs_changed`, and dropped medicines used in less than 1% of patients
- One-hot encoded categorical columns and standardized numeric columns

**Result**

| File | Rows | Columns |
|------|------|---------|
| `cleaned_diabetic_data.csv` | 69,987 | 33 |
| `model_ready_data.csv` | 69,987 | 98 |

Final target: `readmit_30` (1 = readmitted within 30 days, about 7% positive after cleaning).

## Key Findings So Far

- The number of previous inpatient visits is the strongest signal for readmission.
- Patients discharged to a facility or another hospital are readmitted more often than patients discharged home.
- Longer stays, more diagnoses, and more medications are weakly but positively related to readmission.
- Because the target is imbalanced, models should be evaluated with ROC-AUC, recall, and F1, not accuracy alone.

## Repository Structure

```
.
├── data/
│   ├── diabetic_data.csv
│   └── IDS_mapping.csv
├── notebooks/
│   └── EDA_Milestone1.ipynb
├── outputs/
│   ├── cleaned_diabetic_data.csv
│   └── model_ready_data.csv
├── reports/
│   └── Dataset_Exploration_Report.md
└── README.md
```

## How to Run

1. Clone the repository
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. Install the requirements
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Put `diabetic_data.csv` and `IDS_mapping.csv` in the same folder as the notebook, then open it
   ```bash
   jupyter notebook EDA_Milestone1.ipynb
   ```
4. Run all cells. The cleaned datasets will be saved next to the notebook.

## Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter Notebook

## Notes

- The scaler in `model_ready_data.csv` was fit on the full dataset for convenience. In Milestone 3 it will be re-fit on the training split only.
- The dataset is publicly available from the UCI Machine Learning Repository.
