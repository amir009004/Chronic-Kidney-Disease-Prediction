import os
import pickle
import pandas as pd
import numpy as np

# Data Preprocessing & Balancing Tools
# To divide split data into training and testing sets to evaluate unseen data performance
from sklearn.model_selection import train_test_split
# To scale numerical data so model gives equal importance to each value
from sklearn.preprocessing import MinMaxScaler
# To generate synthetic samples and handle severe class imbalance in target variable
from imblearn.over_sampling import SMOTE

# Classification Models to Benchmark
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Model Evaluation Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_and_clean_data(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)

    # Focusing only on the most clinically relevant attributes for prediction
    important_columns = [
        'age', 'bp', 'sg', 'al', 'hemo', 'sc',
        'htn', 'dm', 'cad', 'appet', 'pc', 'classification'
    ]
    df = df[important_columns].copy()

    # Cleaning up trailing spaces & hidden tab characters in string or object columns
    for col in df.select_dtypes(include=['object', 'str']).columns:
        df[col] = df[col].str.strip().str.replace('\t', '', regex=True)

    # Handling missing values by using median for continuous numbers and mode for categorical/discrete fields
    df['age'] = df['age'].fillna(df['age'].median())
    df['bp'] = df['bp'].fillna(df['bp'].median())
    df['sg'] = df['sg'].fillna(df['sg'].mode()[0])
    df['al'] = df['al'].fillna(df['al'].mode()[0])
    df['hemo'] = df['hemo'].fillna(df['hemo'].median())
    df['sc'] = df['sc'].fillna(df['sc'].median())
    df['htn'] = df['htn'].fillna(df['htn'].mode()[0])
    df['dm'] = df['dm'].fillna(df['dm'].mode()[0])
    df['cad'] = df['cad'].fillna(df['cad'].mode()[0])
    df['appet'] = df['appet'].fillna(df['appet'].mode()[0])
    df['pc'] = df['pc'].fillna(df['pc'].mode()[0])

    # Encoding text labels into binary 0 and 1 values for machine learning compatibility
    df['htn'] = df['htn'].map({'yes': 1, 'no': 0})
    df['dm'] = df['dm'].map({'yes': 1, 'no': 0})
    df['cad'] = df['cad'].map({'yes': 1, 'no': 0})
    df['appet'] = df['appet'].map({'good': 1, 'poor': 0})
    df['pc'] = df['pc'].map({'normal': 1, 'abnormal': 0})
    df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})

    return df


def preprocess_and_split(df):
    # Normalizing numerical columns to a 0-1 range to ensure larger values don't dominate the model
    numeric_cols = ['age', 'bp', 'sg', 'al', 'hemo', 'sc']
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    X = df.drop('classification', axis=1)
    y = df['classification']

    # Applying SMOTE to balance the minority class so the model doesn't get biased
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    #  80% training data and 20% held out test data split
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, scaler


def evaluate_models(X_train, X_test, y_train, y_test):
    # Dictionary of various classifiers to benchmark and compare performances
    models = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Classifier": SVC(),
        "Random Forest Classifier": RandomForestClassifier(random_state=42),
        "K Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
    }

    for name, model in models.items():
        print("=" * 50)
        print(f"Benchmarking Model: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def save_artifacts(scaler, model):
    # Automatically creating 'models' folder if it doesn't already exist locally
    os.makedirs("models", exist_ok=True)

    # Serializing the scaler and best model using 'pickle' for future production use
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/model_gbc.pkl", "wb") as f:
        pickle.dump(model, f)
    print("\nSuccessfully saved scaler.pkl and model_gbc.pkl to the 'models/' directory!")


def predict_chronic_disease(age, bp, sg, al, hemo, sc, htn, dm, cad, appet, pc):
    # Loading saved production artifacts to test inference on new patient data
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/model_gbc.pkl", "rb") as f:
        model_gbc = pickle.load(f)

    input_df = pd.DataFrame([{
        'age': age, 'bp': bp, 'sg': sg, 'al': al, 'hemo': hemo, 'sc': sc,
        'htn': htn, 'dm': dm, 'cad': cad, 'appet': appet, 'pc': pc
    }])

    # Mapping raw user inputs to binary values like training data
    input_df['htn'] = input_df['htn'].map({'yes': 1, 'no': 0})
    input_df['dm'] = input_df['dm'].map({'yes': 1, 'no': 0})
    input_df['cad'] = input_df['cad'].map({'yes': 1, 'no': 0})
    input_df['appet'] = input_df['appet'].map({'good': 1, 'poor': 0})
    input_df['pc'] = input_df['pc'].map({'normal': 1, 'abnormal': 0})

    # Applying same fitted MinMaxScaler transformation to the new input
    numeric_cols = ['age', 'bp', 'sg', 'al', 'hemo', 'sc']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    prediction = model_gbc.predict(input_df)
    return "The patient's clinical profile suggests presence of CKD (Chronic Kidney Disease)." if prediction[0] == 1 else "The patient's clinical profile shows no indication of CKD (Chronic Kidney Disease)."


if __name__ == "__main__":
    #  Load and clean the raw dataset
    df_clean = load_and_clean_data("kidney_disease.csv")

    # Normalize features, handle class imbalance through SMOTE, and split data
    X_train, X_test, y_train, y_test, fitted_scaler = preprocess_and_split(df_clean)

    # Run baseline comparisons across multiple ML models
    evaluate_models(X_train, X_test, y_train, y_test)

    # Train top performing model (Gradient Boosting) and save artifacts
    final_model = GradientBoostingClassifier(random_state=42)
    final_model.fit(X_train, y_train)
    save_artifacts(fitted_scaler, final_model)

    # Test inference function with sample patient records
    print("Running Sample Inferences...")
    sample_1 = predict_chronic_disease(30, 80, 1.020, 0.0, 15.4, 1.0, "no", "no", "no", "good", "normal")
    print("Patient 1 Result:", sample_1)

    sample_2 = predict_chronic_disease(65, 160, 1.010, 3.0, 9.0, 3.8, "yes", "yes", "yes", "poor", "abnormal")
    print("Patient 2 Result:", sample_2)