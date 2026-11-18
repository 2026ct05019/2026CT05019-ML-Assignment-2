# Machine Learning Assignment 2

**BITS ID:** 2026CT05019  
**Course:** Machine Learning  
**Task:** Classification model development and comparison

## 1. Problem Statement

The objective is to build and compare five classification models on a public dataset. All models use the same stratified 80:20 train-test split and are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

The models are:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

> The assignment text mentions “6 models” in one place, but the prescribed list and table contain five models. Therefore, the five explicitly specified models are implemented.

## 2. Dataset Description

The **Digits dataset** is used for this multiclass classification problem. It contains handwritten digit images represented by numerical pixel-intensity values.

| Item | Description |
|---|---|
| Dataset | Digits dataset |
| Source | scikit-learn, derived from the UCI handwritten digits dataset |
| Instances | 1,797 |
| Input features | 64 numerical pixel-intensity features |
| Target | Digit class from 0 to 9 |
| Number of classes | 10 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Class distribution | Approximately balanced |
| Train-test split | 80:20, stratified |
| Random state | 5019 |

**Dataset sources:**

- [scikit-learn Digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)

### Preprocessing

1. Checked the dataset for missing values and duplicate rows.
2. Removed duplicate rows before splitting, if present.
3. Used a stratified 80:20 train-test split to maintain class proportions.
4. Used `random_state=5019` for reproducibility.
5. Included `SimpleImputer(strategy="median")` inside every model pipeline. The dataset has no missing values, but this provides explicit leakage-safe missing-value handling.
6. Applied `StandardScaler` to Logistic Regression and kNN because these models are sensitive to feature scale.
7. Did not apply categorical encoding because all 64 input features are numerical.
8. Fitted all learned preprocessing steps only on the training data through scikit-learn pipelines.
9. Exported the holdout test set as `test_data.csv`.

## 3. GitHub Repository Link

**Placeholder:** [Add the final GitHub repository link here](https://github.com/YOUR-USERNAME/2026CT05019-ML-Assignment-2)

Replace the placeholder after creating the repository and verify that the link works while signed out of GitHub.

## 4. Models Used

### Model Configurations

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=3000`, `random_state=5019`; median imputation and standard scaling |
| Decision Tree | `random_state=5019`; median imputation |
| kNN | `n_neighbors=5`, `weights="distance"`; median imputation and standard scaling |
| Naive Bayes | `GaussianNB()`; median imputation |
| Random Forest (Ensemble) | `n_estimators=300`, `random_state=5019`, `n_jobs=-1`; median imputation |

### Evaluation Method

- **Accuracy:** Proportion of correct predictions.
- **AUC:** Macro-average One-vs-Rest AUC calculated from predicted class probabilities.
- **Precision:** Macro average across all ten classes.
- **Recall:** Macro average across all ten classes.
- **F1:** Macro average across all ten classes.
- **MCC:** Multiclass Matthews Correlation Coefficient.

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9667 | 0.9992 | 0.9688 | 0.9664 | 0.9668 | 0.9631 |
| Decision Tree | 0.8389 | 0.9103 | 0.8408 | 0.8385 | 0.8380 | 0.8214 |
| kNN | **0.9750** | 0.9965 | **0.9755** | **0.9748** | **0.9749** | **0.9723** |
| Naive Bayes | 0.8111 | 0.9772 | 0.8436 | 0.8106 | 0.8111 | 0.7939 |
| Random Forest (Ensemble) | 0.9667 | **0.9994** | 0.9677 | 0.9664 | 0.9666 | 0.9631 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Produced a strong linear baseline with 0.9667 Accuracy, 0.9992 AUC, 0.9668 F1, and 0.9631 MCC. Standardization supported stable model fitting, although nonlinear class boundaries limited performance relative to kNN. |
| Decision Tree | Achieved 0.8389 Accuracy, 0.9103 AUC, 0.8380 F1, and 0.8214 MCC. The single tree learned nonlinear patterns but generalized less effectively than Random Forest because of higher variance. |
| kNN | Achieved the highest Accuracy at 0.9750, Precision at 0.9755, Recall at 0.9748, F1 at 0.9749, and MCC at 0.9723. This indicates that samples of the same digit form effective local neighborhoods after standardization. |
| Naive Bayes | Achieved 0.8111 Accuracy, 0.9772 AUC, 0.8111 F1, and 0.7939 MCC. Performance was constrained by Gaussian and conditional-independence assumptions, which are restrictive for correlated image pixels. |
| Random Forest (Ensemble) | Achieved 0.9667 Accuracy, the highest AUC at 0.9994, 0.9666 F1, and 0.9631 MCC. It substantially outperformed the single Decision Tree, demonstrating the variance-reduction benefit of ensemble learning. |
| **Overall Winner for the Dataset** | **kNN** is the overall winner because it achieved the highest Accuracy, Precision, Recall, F1, and MCC. Random Forest achieved the highest AUC, but kNN provided the strongest overall classification performance across the required metrics. |

## 5. Execution Instructions

### Option A: Jupyter Notebook or BITS Virtual Lab

1. Download or clone the repository.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment.
4. Install the required packages.
5. Start Jupyter Notebook.
6. Open `2026CT05019_ML_Assignment_2_Complete.ipynb`.
7. Select **Kernel → Restart Kernel and Run All Cells**.
8. Confirm that all cells complete without errors.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

For Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

### Option B: Google Colab

1. Open Google Colab.
2. Select **File → Upload notebook**.
3. Upload `2026CT05019_ML_Assignment_2.ipynb`.
4. Select **Runtime → Run all**.
5. Download the completed notebook and generated artifacts before the Colab session ends.

A GPU is not required. All five models run on CPU.

## 6. Generated Files

After successful execution, the following artifacts are created:

```text
2026CT05019-ML-Assignment-2/
|-- 2026CT05019_ML_Assignment_2.ipynb
|-- README.md
|-- requirements.txt
|-- test_data.csv
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- random_forest_ensemble.joblib
|-- outputs/
|   |-- model_comparison.csv
|   |-- model_performance_observations.csv
|   |-- all_metrics_comparison.png
|   |-- class_distribution.csv
|   |-- class_distribution.png
|   |-- all_confusion_matrices.png
|   |-- individual confusion-matrix images
|   |-- run_summary.json
|-- screenshots/
|   |-- BITS Virtual Lab execution screenshot
```

## 7. Final Submission Notes

- Execute the final notebook in the BITS Virtual Lab.
- Capture a screenshot showing the Virtual Lab context, notebook name, and successful result output.
- Add the screenshot to the `screenshots` folder.
- Replace the GitHub placeholder in this README and in the report.
- Include this README content in the submitted PDF.
- Confirm that the GitHub repository is accessible and contains all required files.
