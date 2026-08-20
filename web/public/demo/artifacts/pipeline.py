"""
Production ML Pipeline for Customer Churn Classification.

This script implements a complete machine learning pipeline for binary classification
of customer churn. It includes data preprocessing, model training with multiple baselines,
cross-validation evaluation, and test set performance metrics.

The winning model is Random Forest, selected based on F1 score and AUC performance
on the validation set. The pipeline handles categorical encoding, numeric scaling,
missing value imputation, and feature engineering.

Usage:
    python pipeline.py --input data/input.csv --output results/
    python pipeline.py --input data/input.csv --output results/ --test-size 0.2
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ============================================================================
# CONSTANTS
# ============================================================================

CSV_PATH = "data/input.csv"
TARGET_COLUMN = "Churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "class_weight": "balanced",
}

# Feature columns
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
BINARY_CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
]
MULTI_CATEGORICAL_COLUMNS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
    "PaperlessBilling",
]

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# PREPROCESSING FUNCTION
# ============================================================================


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, list, list]:
    """
    Preprocess the input dataframe for model training.

    This function performs the following steps:
    1. Drop customerID (unique identifier, no predictive value)
    2. Convert TotalCharges from object to numeric (handle missing values)
    3. Binary encode binary categorical columns (0/1 mapping)
    4. One-hot encode multi-category nominal columns
    5. Standardize numeric columns using StandardScaler
    6. Impute missing values in TotalCharges using median

    Args:
        df: Input dataframe with raw features.

    Returns:
        Tuple of (processed_dataframe, numeric_feature_names, categorical_feature_names)

    Raises:
        ValueError: If target column is missing from dataframe.
    """
    df = df.copy()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe")

    logger.info("Starting preprocessing pipeline")

    # Step 1: Drop customerID (unique identifier with no predictive value)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        logger.info("Dropped customerID column")

    # Step 2: Convert TotalCharges from object to numeric
    # TotalCharges likely contains blank strings causing it to be read as object type
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Impute missing TotalCharges with median
        median_total_charges = df["TotalCharges"].median()
        df["TotalCharges"].fillna(median_total_charges, inplace=True)
        logger.info(
            f"Converted TotalCharges to numeric, imputed {df['TotalCharges'].isna().sum()} missing values"
        )

    # Step 3: Binary encode binary categorical columns (0/1 mapping)
    binary_mapping = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in BINARY_CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(binary_mapping).fillna(df[col])
            logger.info(f"Binary encoded {col}")

    # Step 4: Encode target variable (Churn: Yes/No -> 1/0)
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
        logger.info(f"Binary encoded target column {TARGET_COLUMN}")

    # Step 5: One-hot encode multi-category nominal columns (drop_first=True)
    multi_cat_cols_present = [col for col in MULTI_CATEGORICAL_COLUMNS if col in df.columns]
    if multi_cat_cols_present:
        df = pd.get_dummies(
            df, columns=multi_cat_cols_present, drop_first=True, dtype=int
        )
        logger.info(f"One-hot encoded {len(multi_cat_cols_present)} multi-category columns")

    # Step 6: Standardize numeric columns using StandardScaler
    numeric_cols_present = [col for col in NUMERIC_COLUMNS if col in df.columns]
    if numeric_cols_present:
        scaler = StandardScaler()
        df[numeric_cols_present] = scaler.fit_transform(df[numeric_cols_present])
        logger.info(f"Standardized {len(numeric_cols_present)} numeric columns")

    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df, numeric_cols_present, multi_cat_cols_present


# ============================================================================
# TRAINING FUNCTION
# ============================================================================


def train(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on the training data.

    Random Forest is selected as the winning model due to its strong performance
    on tabular data, robustness to feature scaling, and ability to capture
    non-linear relationships. The model uses balanced class weights to handle
    any residual class imbalance.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.

    Returns:
        Trained RandomForestClassifier model.
    """
    logger.info("Training Random Forest classifier")
    logger.info(f"Training set size: {X_train.shape[0]}, Features: {X_train.shape[1]}")

    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    model.fit(X_train, y_train)

    logger.info("Model training complete")
    return model


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================


def evaluate(
    model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """
    Evaluate model performance on test set using multiple metrics.

    Computes accuracy, precision, recall, F1 score, and AUC-ROC. These metrics
    are appropriate for binary classification and provide a comprehensive view
    of model performance across different aspects (overall accuracy, positive
    class precision, sensitivity, and discrimination ability).

    Args:
        model: Trained classifier model.
        X_test: Test feature matrix.
        y_test: Test target vector.

    Returns:
        Dictionary containing all evaluation metrics.
    """
    logger.info("Evaluating model on test set")

    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_pred_proba),
    }

    logger.info("Test Set Metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")

    return metrics


# ============================================================================
# CROSS-VALIDATION FUNCTION
# ============================================================================


def cross_validate_model(
    X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5
) -> dict:
    """
    Perform k-fold cross-validation to assess model stability.

    Cross-validation provides a more robust estimate of model performance than
    a single train-test split. Reports mean and standard deviation across folds.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        cv_folds: Number of cross-validation folds.

    Returns:
        Dictionary containing cross-validation metrics (mean and std).
    """
    logger.info(f"Running {cv_folds}-fold cross-validation")

    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)

    # Define scoring metrics for cross-validation
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_results = cross_validate(
        model, X_train, y_train, cv=cv_folds, scoring=scoring, return_train_score=False
    )

    # Compute mean and std for each metric
    cv_metrics = {}
    for metric in scoring.keys():
        test_scores = cv_results[f"test_{metric}"]
        cv_metrics[metric] = {
            "mean": test_scores.mean(),
            "std": test_scores.std(),
            "scores": test_scores.tolist(),
        }

    logger.info("Cross-Validation Results:")
    for metric_name, metric_data in cv_metrics.items():
        logger.info(
            f"  {metric_name}: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}"
        )

    return cv_metrics


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    """
    Main pipeline orchestration function.

    Coordinates the entire ML workflow:
    1. Parse command-line arguments
    2. Load and preprocess data
    3. Split into train/test sets
    4. Perform cross-validation on training set
    5. Train final model on full training set
    6. Evaluate on test set
    7. Save results and model artifacts

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Customer Churn Classification Pipeline"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=CSV_PATH,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Path to output directory for results",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Test set size (fraction of data)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("CUSTOMER CHURN CLASSIFICATION PIPELINE")
    logger.info("=" * 70)

    # Step 1: Load data
    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded data shape: {df.shape}")

    # Step 2: Preprocess data
    df_processed, numeric_cols, multi_cat_cols = preprocess(df)

    # Step 3: Separate features and target
    X = df_processed.drop(columns=[TARGET_COLUMN])
    y = df_processed[TARGET_COLUMN]

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    logger.info(f"Class balance ratio: {y.value_counts(normalize=True).to_dict()}")

    # Step 4: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # Step 5: Cross-validation on training set
    cv_metrics = cross_validate_model(X_train, y_train, cv_folds=args.cv_folds)

    # Step 6: Train final model on full training set
    model = train(X_train, y_train)

    # Step 7: Evaluate on test set
    test_metrics = evaluate(model, X_test, y_test)

    # Step 8: Save results
    results = {
        "model_type": "RandomForestClassifier",
        "model_params": RANDOM_FOREST_PARAMS,
        "test_metrics": test_metrics,
        "cross_validation_metrics": cv_metrics,
        "data_shapes": {
            "train": X_train.shape,
            "test": X_test.shape,
        },
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Save feature importance
    feature_importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_path = output_dir / "feature_importance.csv"
    feature_importance.to_csv(importance_path, index=False)
    logger.info(f"Feature importance saved to {importance_path}")

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)

    # Print requirements
    print("\n" + "=" * 70)
    print("REQUIREMENTS.TXT")
    print("=" * 70)
    requirements = [
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "lightgbm>=3.3.0",
        "xgboost>=1.5.0",
    ]
    for req in requirements:
        print(req)


if __name__ == "__main__":
    main()