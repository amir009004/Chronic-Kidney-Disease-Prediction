# Chronic Kidney Disease (CKD) Diagnostic Classifier

An end-to-end machine learning pipeline built to evaluate clinical patient biomarkers and classify whether a patient currently has Chronic Kidney Disease (`ckd`) or not (`notckd`). It is designed to assist healthcare providers and clinical decision systems in flagging early indicators from routine lab results.

---

## Background

Chronic Kidney Disease (CKD) affects an estimated 788 million adults worldwide each year according to a 2025 Lancet analysis, which makes it a significant global health concern. In the United States, approximately 37 million adults (14%) suffer from CKD, with most unaware of their condition, according to the CDC's 2026 estimates. Early-stage CKD often presents without noticeable symptoms so laboratory measurements play the main role in identifying kidney dysfunction. This project explores how machine learning can use clinical indicators such as serum creatinine, blood urea, hemoglobin, blood pressure, and urine specific gravity to classify CKD status and support data-driven approaches to earlier disease identification.

## Dataset

* **Source:** UCI Machine Learning Repository (Chronic Kidney Disease Data Set)
* **Records:** 400 patient instances
* **Attributes:** 25 clinical and laboratory features (including age, blood pressure, specific gravity, albumin, hemoglobin, serum creatinine, hypertension, diabetes mellitus, coronary artery disease, appetite, and pus cells).

| Feature Category | Key Variables Included | Description / Clinical Role |
| :--- | :--- | :--- |
| **Demographic & Vitals** | `age`, `bp` (Blood Pressure) | General patient vitals and baseline metrics |
| **Urinalysis & Chemistry** | `sg` (Specific Gravity), `al` (Albumin), `sc` (Serum Creatinine), `hemo` (Hemoglobin) | Core laboratory markers indicating renal filtration efficiency |
| **Comorbidities & Symptoms** | `htn` (Hypertension), `dm` (Diabetes), `cad` (Coronary Artery Disease), `appet`, `pc` | Patient clinical history and physical symptoms |
| **Target Label** | `classification` | Binary diagnosis (`ckd` = 1, `notckd` = 0) |

---

## Methodology & Pipeline Architecture

1. **Data Cleaning & Preprocessing:** 
   * Filtered down to the 11 most clinically relevant biomarkers.
   * Cleaned trailing whitespace and hidden tab characters from string fields.
   * Handled missing values (`NaN`) robustly using **median imputation** for continuous numerical features and **mode imputation** for discrete/categorical fields.
   * Mapped text labels (`yes`/`no`, `good`/`poor`, `normal`/`abnormal`, `ckd`/`notckd`) into binary integers (`0` and `1`).
2. **Feature Scaling:** Normalized numerical columns using scikit-learn's `MinMaxScaler` to ensure equal feature weighting.
3. **Class Balancing (SMOTE):** Applied Synthetic Minority Over-sampling Technique (`SMOTE`) on the training split to eliminate class bias and handle data skew.
4. **Model Benchmarking:** Evaluated 8 distinct machine learning classifiers using stratified train-test splits (80/20).

---

## Results & Model Performance

Several classification algorithms were benchmarked to identify the most robust architecture for production inference. **Gradient Boosting** achieved top-tier performance across accuracy and classification metrics.

| Model | Accuracy | Precision (CKD) | Recall (CKD) | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting Classifier** | **~99%** | **0.99** | **0.99** | **0.99** |
| Random Forest Classifier | ~98% | 0.98 | 0.98 | 0.98 |
| Support Vector Classifier (SVC) | ~97% | 0.97 | 0.97 | 0.97 |
| Logistic Regression | ~96% | 0.96 | 0.96 | 0.96 |
| K-Nearest Neighbors (KNN) | ~94% | 0.94 | 0.94 | 0.94 |

> **Clinical Validation Note:** Gradient Boosting demonstrated exceptional sensitivity in identifying positive CKD indicators, minimizing false negatives which is critical in medical screening contexts.

---

## Key Findings

* **Biomarker Importance:** Serum creatinine (`sc`), hemoglobin (`hemo`), and specific gravity (`sg`) emerged as the strongest physiological predictors of chronic kidney disease.
* **Impact of SMOTE:** Applying over-sampling drastically improved the model's ability to capture minority-class patterns without overfitting.
* **Production Serialization:** The finalized model and scaler were successfully serialized using `pickle` into binary artifacts (`model_gbc.pkl`, `scaler.pkl`) to support instant, stateless real-time patient inference.

---

## Citation

* University of California, Irvine. "Chronic Kidney Disease." UCI Machine Learning Repository. [UCI Machine Learning Repository - Chronic Kidney Disease] https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease
* Global Burden of Disease 2023 Chronic Kidney Disease Collaborators. “Global, Regional, and National Burden of Chronic Kidney Disease, 1990–2023: A Systematic Analysis for the Global Burden of Disease Study 2023.” The Lancet (2025). https://www.sciencedirect.com/science/article/pii/S0140673625018537
* Centers for Disease Control and Prevention. “Chronic Kidney Disease in the United States.” March 31, 2026. https://www.cdc.gov/kidney-disease/php/data-research/

## Project File Structure

```text
CKD-Prediction-Project/
│
├── data/
│   └── kidney_disease.csv      # Raw UCI CKD dataset
├── models/
│   ├── model_gbc.pkl           # Serialized Gradient Boosting production model
│   └── scaler.pkl              # Fitted MinMaxScaler object for input normalization
├── train_model.py              # Main Python pipeline script
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
