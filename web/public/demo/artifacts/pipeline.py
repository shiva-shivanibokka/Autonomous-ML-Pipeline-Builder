"""
Production ML Pipeline for Customer Churn Classification.

This script implements a complete machine learning pipeline for binary classification
of customer churn. It includes data preprocessing, model training with cross-validation,
evaluation on a held-out test set, and comparison of multiple baseline models.

The pipeline:
1. Loads and preprocesses the input CSV (handles type conversions, encoding, scaling)
2. Trains multiple models (Random Forest, Logistic Regression, LightGBM, XGBoost)
3. Evaluates each model using cross-validation and test set metrics
4. Selects the best model based on F1 score (primary metric for imbalanced classification)
5. Logs results and generates a final evaluation report

Usage:
    python pipeline.py --input data.csv --output results.csv --seed 42
"""

import argparse
import logging
import sys
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report
)
import lightgbm as lgb
import xgboost as xgb

# ============================================================================
# CONSTANTS
# ============================================================================

CSV_PATH = "data.csv"
TARGET_COLUMN = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Numeric columns to scale
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges"]

# Binary categorical columns (Yes/No) to label encode
BINARY_CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService"]

# Multi-class categorical columns to one-hot encode
MULTICLASS_CATEGORICAL = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection"
]

# Columns to drop (unique identifiers, no predictive value)
DROP_COLUMNS = ["customerID"]

# Model hyperparameters
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "class_weight": "balanced"
}

LGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 7,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "class_weight": "balanced"
}

XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 7,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "scale_pos_weight": 1.0
}

LR_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "class_weight": "balanced"
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# PREPROCESSING FUNCTION
# ============================================================================

def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Preprocess the input dataframe for model training.

    Steps:
    1. Drop unique identifiers (customerID)
    2. Remove duplicate rows
    3. Convert TotalCharges from object to numeric (handle blanks)
    4. Label encode binary Yes/No columns
    5. One-hot encode multi-class categorical columns
    6. Encode target variable (Churn: Yes=1, No=0)
    7. Scale numeric features using StandardScaler
    8. Handle SeniorCitizen as binary (no scaling needed)

    Args:
        df: Input dataframe with raw features.

    Returns:
        Tuple of (preprocessed dataframe, fitted StandardScaler).

    Raises:
        ValueError: If target column is missing or preprocessing fails.
    """
    logger.info("Starting preprocessing...")
    df = df.copy()

    # Drop unique identifiers
    for col in DROP_COLUMNS:
        if col in df.columns:
            logger.info(f"Dropping column: {col}")
            df = df.drop(columns=[col])

    # Remove duplicate rows (common in Telco churn dataset)
    initial_rows = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_rows - len(df)} duplicate rows")

    # Convert TotalCharges from object to numeric
    if "TotalCharges" in df.columns:
        logger.info("Converting TotalCharges to numeric...")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Impute NaNs with 0 (likely customers with tenure=0)
        nan_count = df["TotalCharges"].isna().sum()
        if nan_count > 0:
            logger.info(f"Imputing {nan_count} NaN values in TotalCharges with 0")
            df["TotalCharges"].fillna(0, inplace=True)

    # Label encode binary Yes/No columns (0/1)
    binary_mapping = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in BINARY_CATEGORICAL:
        if col in df.columns:
            logger.info(f"Label encoding binary column: {col}")
            df[col] = df[col].map(binary_mapping)

    # One-hot encode multi-class categorical columns
    for col in MULTICLASS_CATEGORICAL:
        if col in df.columns:
            logger.info(f"One-hot encoding column: {col}")
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])

    # Encode target variable
    if TARGET_COLUMN in df.columns:
        logger.info(f"Encoding target column: {TARGET_COLUMN}")
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    # Identify numeric columns for scaling (exclude target and binary encoded)
    numeric_cols_to_scale = [col for col in NUMERIC_COLUMNS if col in df.columns]
    if "TotalCharges" in df.columns:
        numeric_cols_to_scale.append("TotalCharges")

    # Scale numeric features
    logger.info(f"Scaling numeric columns: {numeric_cols_to_scale}")
    scaler = StandardScaler()
    df[numeric_cols_to_scale] = scaler.fit_transform(df[numeric_cols_to_scale])

    # SeniorCitizen is binary (0/1), no scaling needed
    logger.info("SeniorCitizen treated as binary categorical (no scaling)")

    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df, scaler


# ============================================================================
# TRAIN FUNCTION
# ============================================================================

def train(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """
    Train multiple models and return the best one based on F1 score.

    Models trained:
    - Random Forest (primary model per requirements)
    - Logistic Regression (simple baseline)
    - LightGBM (gradient boosting baseline)
    - XGBoost (gradient boosting baseline)

    Each model is evaluated using stratified 5-fold cross-validation.
    The model with the highest mean CV F1 score is selected.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.

    Returns:
        Dictionary containing:
        - 'model': Trained model object
        - 'model_name': Name of the best model
        - 'cv_scores': Cross-validation scores for all models
        - 'scaler': StandardScaler fitted on training data
    """
    logger.info("Starting model training...")

    # Initialize models
    models = {
        "RandomForest": RandomForestClassifier(**RF_PARAMS),
        "LogisticRegression": LogisticRegression(**LR_PARAMS),
        "LightGBM": lgb.LGBMClassifier(**LGB_PARAMS),
        "XGBoost": xgb.XGBClassifier(**XGB_PARAMS)
    }

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc"
    }

    cv_results = {}
    best_model_name = None
    best_f1_mean = -np.inf

    # Train and evaluate each model
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        cv_scores = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
        )
        cv_results[model_name] = cv_scores

        # Calculate mean F1 score (primary metric for classification)
        f1_mean = cv_scores["test_f1"].mean()
        f1_std = cv_scores["test_f1"].std()
        logger.info(
            f"{model_name} - F1: {f1_mean:.4f} ± {f1_std:.4f}, "
            f"AUC: {cv_scores['test_roc_auc'].mean():.4f}"
        )

        # Select best model based on F1 score
        if f1_mean > best_f1_mean:
            best_f1_mean = f1_mean
            best_model_name = model_name

    # Train the best model on full training set
    logger.info(f"Best model: {best_model_name} (F1: {best_f1_mean:.4f})")
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)

    return {
        "model": best_model,
        "model_name": best_model_name,
        "cv_scores": cv_results
    }


# ============================================================================
# EVALUATE FUNCTION
# ============================================================================

def evaluate(model: Any, X_test: pd.DataFrame, y_test: pd.Series,
             model_name: str = "Model") -> Dict[str, float]:
    """
    Evaluate model performance on test set.

    Computes:
    - Accuracy: Overall correctness
    - F1 Score: Harmonic mean of precision and recall (primary metric)
    - Precision: True positives / (true positives + false positives)
    - Recall: True positives / (true positives + false negatives)
    - ROC-AUC: Area under the receiver operating characteristic curve

    Args:
        model: Trained model object with predict() and predict_proba() methods.
        X_test: Test feature matrix.
        y_test: Test target vector.
        model_name: Name of the model for logging.

    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info(f"Evaluating {model_name} on test set...")

    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred_proba)
    }

    # Log results
    logger.info(f"{model_name} Test Set Results:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")

    # Log confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    return metrics


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main pipeline orchestration function.

    Workflow:
    1. Parse command-line arguments
    2. Load input CSV
    3. Preprocess data (encoding, scaling, feature engineering)
    4. Split into train/test sets (stratified)
    5. Train multiple models with cross-validation
    6. Evaluate best model on test set
    7. Save results and model artifacts
    """
    parser = argparse.ArgumentParser(
        description="ML Pipeline for Customer Churn Classification"
    )
    parser.add_argument(
        "--input", type=str, default=CSV_PATH,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output", type=str, default="results.csv",
        help="Path to output results CSV"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_STATE,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("CUSTOMER CHURN CLASSIFICATION PIPELINE")
    logger.info("=" * 80)

    # Load data
    logger.info(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Preprocess
    df_processed, scaler = preprocess(df)

    # Separate features and target
    if TARGET_COLUMN not in df_processed.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in data")

    X = df_processed.drop(columns=[TARGET_COLUMN])
    y = df_processed[TARGET_COLUMN]

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    logger.info(f"Class balance: {y.value_counts(normalize=True)}")

    # Train/test split (stratified to preserve class distribution)
    logger.info(f"Splitting data: {100 * (1 - TEST_SIZE):.0f}% train, "
                f"{100 * TEST_SIZE:.0f}% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=args.seed, stratify=y
    )
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Train models
    training_results = train(X_train, y_train)
    best_model = training_results["model"]
    best_model_name = training_results["model_name"]
    cv_scores = training_results["cv_scores"]

    # Evaluate on test set
    test_metrics = evaluate(best_model, X_test, y_test, best_model_name)

    # Log cross-validation results
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-VALIDATION RESULTS (5-Fold Stratified)")
    logger.info("=" * 80)
    for model_name, scores in cv_scores.items():
        logger.info(f"\n{model_name}:")
        for metric in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
            key = f"test_{metric}"
            mean = scores[key].mean()
            std = scores[key].std()
            logger.info(f"  {metric}: {mean:.4f} ± {std:.4f}")

    # Log final results
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"Test Set Metrics:")
    for metric_name, metric_value in test_metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")

    # Save results to CSV
    results_df = pd.DataFrame([{
        "model": best_model_name,
        **test_metrics
    }])
    results_df.to_csv(args.output, index=False)
    logger.info(f"\nResults saved to {args.output}")

    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

    # Print requirements.txt content
    print("\n" + "=" * 80)
    print("REQUIREMENTS.TXT")
    print("=" * 80)
    requirements = """pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
xgboost>=1.5.0"""
    print(requirements)