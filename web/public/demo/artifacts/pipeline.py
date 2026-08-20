"""
Production ML Pipeline for Customer Churn Classification.

This script implements a complete machine learning pipeline for binary classification
of customer churn. It includes data preprocessing, feature engineering, model training
with cross-validation, and comprehensive evaluation metrics.

The pipeline:
1. Loads and preprocesses customer data
2. Applies feature engineering (tenure buckets, interaction features)
3. Trains multiple models (RandomForest, LightGBM, XGBoost) with cross-validation
4. Selects the best model based on F1 score (primary metric for imbalanced classification)
5. Evaluates on held-out test set with detailed metrics
6. Logs results for reproducibility

Target: Binary classification of 'Churn' (Yes/No)
Primary Metric: F1 score (handles class imbalance better than accuracy)
"""

import argparse
import logging
import sys
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report
)
import lightgbm as lgb
import xgboost as xgb

# ============================================================================
# CONSTANTS
# ============================================================================

CSV_PATH = "data/churn.csv"
TARGET_COLUMN = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature columns
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges"]
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection"
]
BINARY_CATEGORICAL_COLS = ["gender", "Partner", "Dependents", "PhoneService"]
MULTI_CATEGORICAL_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection"
]

# Model hyperparameters
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

LGBM_PARAMS = {
    "n_estimators": 100,
    "max_depth": 7,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "verbose": -1
}

XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "scale_pos_weight": 1.0,  # Will be adjusted based on class imbalance
    "random_state": RANDOM_STATE,
    "verbosity": 0
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# PREPROCESSING FUNCTION
# ============================================================================

def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Preprocess customer churn data with feature engineering.

    Args:
        df: Raw input DataFrame with customer data.

    Returns:
        Tuple of (processed DataFrame, preprocessing metadata dict).

    Preprocessing steps:
        1. Drop customerID (unique identifier, no predictive value)
        2. Convert TotalCharges from object to numeric (handle blanks)
        3. Impute missing TotalCharges using tenure * MonthlyCharges
        4. Apply label encoding to binary categorical columns
        5. Apply one-hot encoding to multi-level categorical columns
        6. Create tenure buckets (feature engineering)
        7. Create interaction feature (MonthlyCharges * tenure)
        8. Scale numeric features using StandardScaler
        9. Validate dtypes and handle any remaining issues
    """
    logger.info("Starting preprocessing...")
    df = df.copy()
    metadata = {}

    # Step 1: Drop customerID (unique identifier with no predictive value)
    if "customerID" in df.columns:
        logger.info("Dropping customerID column (unique identifier)")
        df = df.drop(columns=["customerID"])

    # Step 2: Convert TotalCharges from object to numeric
    if "TotalCharges" in df.columns:
        logger.info("Converting TotalCharges to numeric (handling blanks)")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Step 3: Impute missing TotalCharges
        # For new customers (tenure=0), TotalCharges should be 0
        # Otherwise, estimate as tenure * MonthlyCharges
        missing_mask = df["TotalCharges"].isna()
        if missing_mask.sum() > 0:
            logger.info(f"Imputing {missing_mask.sum()} missing TotalCharges values")
            df.loc[missing_mask, "TotalCharges"] = (
                df.loc[missing_mask, "tenure"] * df.loc[missing_mask, "MonthlyCharges"]
            )

    # Step 4: Encode target variable (Churn)
    if TARGET_COLUMN in df.columns:
        logger.info(f"Encoding target column '{TARGET_COLUMN}'")
        le_target = LabelEncoder()
        df[TARGET_COLUMN] = le_target.fit_transform(df[TARGET_COLUMN])
        metadata["target_encoder"] = le_target
        logger.info(f"Target classes: {le_target.classes_}")

    # Step 5: Apply label encoding to binary categorical columns
    label_encoders = {}
    for col in BINARY_CATEGORICAL_COLS:
        if col in df.columns:
            logger.info(f"Label encoding binary column '{col}'")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    metadata["label_encoders"] = label_encoders

    # Step 6: Apply one-hot encoding to multi-level categorical columns
    # Use drop_first=True to avoid multicollinearity
    ohe_cols = []
    for col in MULTI_CATEGORICAL_COLS:
        if col in df.columns:
            logger.info(f"One-hot encoding multi-level column '{col}'")
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            ohe_cols.extend(dummies.columns.tolist())
            df = df.drop(columns=[col])

    metadata["ohe_columns"] = ohe_cols

    # Step 7: Feature engineering - tenure buckets
    if "tenure" in df.columns:
        logger.info("Creating tenure bucket features")
        df["tenure_0_12"] = (df["tenure"] >= 0) & (df["tenure"] < 12)
        df["tenure_12_24"] = (df["tenure"] >= 12) & (df["tenure"] < 24)
        df["tenure_24_48"] = (df["tenure"] >= 24) & (df["tenure"] < 48)
        df["tenure_48_plus"] = df["tenure"] >= 48
        # Convert boolean to int
        for col in ["tenure_0_12", "tenure_12_24", "tenure_24_48", "tenure_48_plus"]:
            df[col] = df[col].astype(int)

    # Step 8: Feature engineering - interaction feature
    if "MonthlyCharges" in df.columns and "tenure" in df.columns:
        logger.info("Creating interaction feature: MonthlyCharges * tenure")
        df["charges_tenure_interaction"] = df["MonthlyCharges"] * df["tenure"]

    # Step 9: Scale numeric features
    # SeniorCitizen is binary (0/1), so we don't scale it
    numeric_to_scale = ["tenure", "MonthlyCharges"]
    if "TotalCharges" in df.columns:
        numeric_to_scale.append("TotalCharges")
    if "charges_tenure_interaction" in df.columns:
        numeric_to_scale.append("charges_tenure_interaction")

    logger.info(f"Scaling numeric features: {numeric_to_scale}")
    scaler = StandardScaler()
    df[numeric_to_scale] = scaler.fit_transform(df[numeric_to_scale])
    metadata["scaler"] = scaler
    metadata["scaled_columns"] = numeric_to_scale

    # Step 10: Validate dtypes
    logger.info("Validating dtypes...")
    for col in df.columns:
        if df[col].dtype == "object":
            logger.warning(f"Column '{col}' still has object dtype after preprocessing")

    logger.info(f"Preprocessing complete. Shape: {df.shape}")
    return df, metadata


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "random_forest"
) -> Tuple[Any, Dict[str, float]]:
    """
    Train multiple models with cross-validation and select the best one.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        model_name: Name of model to train ('random_forest', 'lightgbm', 'xgboost', 'all').

    Returns:
        Tuple of (trained model, cross-validation metrics dict).

    The function trains the specified model(s) using stratified k-fold cross-validation
    and reports mean ± std for F1 score (primary metric for classification).
    """
    logger.info(f"Training model(s): {model_name}")

    # Calculate class weights for imbalanced data
    n_samples = len(y_train)
    n_positive = (y_train == 1).sum()
    n_negative = (y_train == 0).sum()
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
    logger.info(f"Class distribution - Negative: {n_negative}, Positive: {n_positive}")
    logger.info(f"Scale pos weight for XGBoost: {scale_pos_weight:.2f}")

    # Setup cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc"
    }

    models = {}
    cv_results = {}

    # Train Random Forest
    if model_name in ["random_forest", "all"]:
        logger.info("Training Random Forest...")
        rf = RandomForestClassifier(**RF_PARAMS)
        rf_cv = cross_validate(rf, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        models["random_forest"] = rf
        cv_results["random_forest"] = rf_cv
        logger.info(
            f"RF CV F1: {rf_cv['test_f1'].mean():.4f} ± {rf_cv['test_f1'].std():.4f}"
        )

    # Train LightGBM
    if model_name in ["lightgbm", "all"]:
        logger.info("Training LightGBM...")
        lgbm = lgb.LGBMClassifier(**LGBM_PARAMS)
        lgbm_cv = cross_validate(lgbm, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        models["lightgbm"] = lgbm
        cv_results["lightgbm"] = lgbm_cv
        logger.info(
            f"LGBM CV F1: {lgbm_cv['test_f1'].mean():.4f} ± {lgbm_cv['test_f1'].std():.4f}"
        )

    # Train XGBoost
    if model_name in ["xgboost", "all"]:
        logger.info("Training XGBoost...")
        xgb_params = XGBOOST_PARAMS.copy()
        xgb_params["scale_pos_weight"] = scale_pos_weight
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_cv = cross_validate(xgb_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        models["xgboost"] = xgb_model
        cv_results["xgboost"] = xgb_cv
        logger.info(
            f"XGB CV F1: {xgb_cv['test_f1'].mean():.4f} ± {xgb_cv['test_f1'].std():.4f}"
        )

    # Select best model based on F1 score (primary metric for classification)
    best_model_name = max(
        cv_results.keys(),
        key=lambda k: cv_results[k]["test_f1"].mean()
    )
    logger.info(f"Best model selected: {best_model_name}")

    # Train the best model on full training set
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)

    # Prepare metrics summary
    metrics_summary = {
        f"{best_model_name}_f1_mean": cv_results[best_model_name]["test_f1"].mean(),
        f"{best_model_name}_f1_std": cv_results[best_model_name]["test_f1"].std(),
        f"{best_model_name}_accuracy_mean": cv_results[best_model_name]["test_accuracy"].mean(),
        f"{best_model_name}_roc_auc_mean": cv_results[best_model_name]["test_roc_auc"].mean(),
    }

    return best_model, metrics_summary


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "random_forest"
) -> Dict[str, float]:
    """
    Evaluate model on held-out test set with comprehensive metrics.

    Args:
        model: Trained model object.
        X_test: Test feature matrix.
        y_test: Test target vector.
        model_name: Name of the model (for logging).

    Returns:
        Dictionary of evaluation metrics.

    Metrics computed:
        - Accuracy: Overall correctness
        - F1 Score: Harmonic mean of precision and recall (primary metric)
        - Precision: True positives / (true positives + false positives)
        - Recall: True positives / (true positives + false negatives)
        - ROC-AUC: Area under the ROC curve
        - Confusion Matrix: TP, TN, FP, FN
    """
    logger.info(f"Evaluating {model_name} on test set...")

    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Log results
    logger.info(f"Test Set Metrics for {model_name}:")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  ROC-AUC:   {auc:.4f}")
    logger.info(f"  Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # Feature importance (if available)
    if hasattr(model, "feature_importances_"):
        logger.info("Top 10 Most Important Features:")
        feature_importance = pd.DataFrame({
            "feature": X_test.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }

    return metrics


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main pipeline orchestration function.

    Parses command-line arguments, loads data, preprocesses, trains models,
    and evaluates on test set. Logs all results for reproducibility.
    """
    parser = argparse.ArgumentParser(
        description="Customer Churn Classification Pipeline"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=CSV_PATH,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "lightgbm", "xgboost", "all"],
        help="Model to train"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Test set size (0.0-1.0)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("CUSTOMER CHURN CLASSIFICATION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"CSV Path: {args.csv_path}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Test Size: {args.test_size}")
    logger.info(f"Random State: {args.random_state}")

    # Load data
    logger.info("\nLoading data...")
    try:
        df = pd.read_csv(args.csv_path)
        logger.info(f"Data loaded. Shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"File not found: {args.csv_path}")
        sys.exit(1)

    # Preprocess
    logger.info("\n" + "=" * 70)
    logger.info("PREPROCESSING")
    logger.info("=" * 70)
    df_processed, metadata = preprocess(df)

    # Separate features and target
    X = df_processed.drop(columns=[TARGET_COLUMN])
    y = df_processed[TARGET_COLUMN]

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")

    # Stratified train-test split (preserve class proportions)
    logger.info("\nPerforming stratified train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Train
    logger.info("\n" + "=" * 70)
    logger.info("MODEL TRAINING")
    logger.info("=" * 70)
    model, cv_metrics = train(X_train, y_train, model_name=args.model)

    logger.info("\nCross-Validation Metrics:")
    for metric_name, metric_value in cv_metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")

    # Evaluate
    logger.info("\n" + "=" * 70)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 70)
    test_metrics = evaluate(model, X_test, y_test, model_name=args.model)

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)

    return model, metadata, test_metrics


if __name__ == "__main__":
    main()

    # Print requirements.txt content
    print("\n" + "=" * 70)
    print("REQUIREMENTS.TXT")
    print("=" * 70)
    requirements = """pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
xgboost>=1.5.0"""
    print(requirements)