"""
model.py — Train and evaluate climb grade prediction models.

Primary model: XGBoost regressor predicting the community difficulty_average
on the Kilter Board's internal scale (0–40), which maps to V-grades.

Evaluation:
  - MAE in V-grade units (target: < 1.0)
  - Within-1-grade accuracy
  - Within-2-grade accuracy
  - Per-grade MAE breakdown
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Using RandomForest as fallback.")

from feature_engineering import FEATURE_COLUMNS


# --- Grade utilities ---

# Map internal difficulty scale to V-grade number for evaluation
def difficulty_to_v_grade(difficulty: float) -> float:
    """Convert internal difficulty scale to approximate V-grade number.
    
    Internal scale: ~10 = V0, ~40 = V15
    Roughly: V = (difficulty - 10) / 2
    """
    return max(0, (difficulty - 10) / 2)


def v_grade_to_difficulty(v_grade: float) -> float:
    """Convert V-grade number back to internal difficulty scale."""
    return v_grade * 2 + 10


# --- Model training ---

def train_xgboost(X_train, y_train, X_val, y_val) -> "xgb.XGBRegressor":
    """Train an XGBoost regressor with early stopping.
    
    Args:
        X_train, y_train: Training features and targets
        X_val, y_val: Validation features and targets
        
    Returns:
        Trained XGBRegressor
    """
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric="mae",
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    
    return model


def train_random_forest(X_train, y_train) -> RandomForestRegressor:
    """Train a Random Forest regressor as baseline/fallback.
    
    Args:
        X_train, y_train: Training features and targets
        
    Returns:
        Trained RandomForestRegressor
    """
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=10,
        min_samples_split=20,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    
    model.fit(X_train, y_train)
    return model


def train_ridge_baseline(X_train, y_train) -> Ridge:
    """Train a Ridge regression as a simple baseline.
    
    Args:
        X_train, y_train: Training features and targets
        
    Returns:
        Trained Ridge model
    """
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model


# --- Evaluation ---

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """Evaluate a model on test data.
    
    Args:
        model: Trained model with .predict() method
        X_test: Test features
        y_test: Test targets (internal difficulty scale)
        model_name: Name for display
        
    Returns:
        Dict of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    # Metrics on internal scale
    mae_internal = mean_absolute_error(y_test, y_pred)
    rmse_internal = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Convert to V-grade scale for interpretable metrics
    y_test_v = np.array([difficulty_to_v_grade(d) for d in y_test])
    y_pred_v = np.array([difficulty_to_v_grade(d) for d in y_pred])
    
    mae_v = mean_absolute_error(y_test_v, y_pred_v)
    
    # Within-N-grade accuracy
    abs_errors_v = np.abs(y_test_v - y_pred_v)
    within_1 = (abs_errors_v <= 1.0).mean()
    within_2 = (abs_errors_v <= 2.0).mean()
    exact = (abs_errors_v <= 0.5).mean()
    
    metrics = {
        "model_name": model_name,
        "mae_internal": round(mae_internal, 3),
        "rmse_internal": round(rmse_internal, 3),
        "r2": round(r2, 4),
        "mae_v_grade": round(mae_v, 3),
        "within_half_grade": round(exact, 4),
        "within_1_grade": round(within_1, 4),
        "within_2_grades": round(within_2, 4),
        "n_test": len(y_test),
    }
    
    print(f"\n{'='*50}")
    print(f"  {model_name} Results")
    print(f"{'='*50}")
    print(f"  MAE (internal scale): {mae_internal:.3f}")
    print(f"  RMSE (internal scale): {rmse_internal:.3f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE (V-grade): {mae_v:.3f}")
    print(f"  Within ½ grade: {exact:.1%}")
    print(f"  Within 1 grade: {within_1:.1%}")
    print(f"  Within 2 grades: {within_2:.1%}")
    print(f"  Test samples: {len(y_test)}")
    
    return metrics


def get_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Extract feature importances from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        
    Returns:
        DataFrame sorted by importance
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    
    importance_df["importance_pct"] = (
        importance_df["importance"] / importance_df["importance"].sum() * 100
    ).round(2)
    
    return importance_df


def per_grade_analysis(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    grade_labels: pd.Series,
) -> pd.DataFrame:
    """Analyse model performance broken down by grade.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        grade_labels: V-grade labels for test set
        
    Returns:
        DataFrame with per-grade metrics
    """
    y_pred = model.predict(X_test)
    
    results = []
    for grade in sorted(grade_labels.unique()):
        mask = grade_labels == grade
        if mask.sum() < 5:
            continue
        
        grade_mae = mean_absolute_error(y_test[mask], y_pred[mask])
        grade_mae_v = grade_mae / 2  # Approximate V-grade conversion
        
        results.append({
            "v_grade": grade,
            "n_samples": int(mask.sum()),
            "mae_internal": round(grade_mae, 3),
            "mae_v_approx": round(grade_mae_v, 3),
        })
    
    return pd.DataFrame(results)


# --- Main pipeline ---

def prepare_data(features_df: pd.DataFrame):
    """Prepare train/val/test splits from feature DataFrame.
    
    If the data has a 'split' column (from HuggingFace dataset), use those.
    Otherwise, create splits from scratch.
    
    Args:
        features_df: DataFrame with features and targets
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, test_meta)
    """
    target_col = "difficulty_average"
    
    if "split" in features_df.columns:
        train_df = features_df[features_df["split"] == "train"]
        val_df = features_df[features_df["split"] == "val"]
        test_df = features_df[features_df["split"] == "test"]
    else:
        # Split by UUID so same climb at different angles stays together
        uuids = features_df["uuid"].unique()
        train_uuids, temp_uuids = train_test_split(uuids, test_size=0.2, random_state=42)
        val_uuids, test_uuids = train_test_split(temp_uuids, test_size=0.5, random_state=42)
        
        train_df = features_df[features_df["uuid"].isin(train_uuids)]
        val_df = features_df[features_df["uuid"].isin(val_uuids)]
        test_df = features_df[features_df["uuid"].isin(test_uuids)]
    
    X_train = train_df[FEATURE_COLUMNS].values
    X_val = val_df[FEATURE_COLUMNS].values
    X_test = test_df[FEATURE_COLUMNS].values
    
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values
    
    test_meta = test_df[["uuid", "v_grade", "v_grade_numeric", "ascensionist_count"]].copy()
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, test_meta


def main():
    parser = argparse.ArgumentParser(description="Train grade prediction model")
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to features directory or parquet file"
    )
    parser.add_argument(
        "--output", type=str, default="models",
        help="Output directory for saved models"
    )
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load features
    if input_path.is_file():
        features_df = pd.read_parquet(input_path)
    else:
        features_df = pd.read_parquet(input_path / "features.parquet")
    
    print(f"Loaded {len(features_df)} samples with {len(FEATURE_COLUMNS)} features")
    
    # Prepare splits
    X_train, X_val, X_test, y_train, y_val, y_test, test_meta = prepare_data(features_df)
    
    # --- Train models ---
    all_metrics = []
    
    # 1. Ridge baseline
    print("\nTraining Ridge baseline...")
    ridge = train_ridge_baseline(X_train, y_train)
    ridge_metrics = evaluate_model(ridge, X_test, y_test, "Ridge (baseline)")
    all_metrics.append(ridge_metrics)
    
    # 2. Random Forest
    print("\nTraining Random Forest...")
    rf = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test, "Random Forest")
    all_metrics.append(rf_metrics)
    
    # 3. XGBoost (primary)
    if HAS_XGBOOST:
        print("\nTraining XGBoost...")
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        all_metrics.append(xgb_metrics)
        best_model = xgb_model
        best_name = "XGBoost"
    else:
        best_model = rf
        best_name = "Random Forest"
    
    # --- Feature importance ---
    importance = get_feature_importance(best_model, FEATURE_COLUMNS)
    if not importance.empty:
        print(f"\nTop 15 features ({best_name}):")
        print(importance.head(15).to_string(index=False))
        importance.to_csv(output_path / "feature_importance.csv", index=False)
    
    # --- Per-grade analysis ---
    grade_analysis = per_grade_analysis(
        best_model, X_test, y_test, test_meta["v_grade"]
    )
    if not grade_analysis.empty:
        print(f"\nPer-grade MAE:")
        print(grade_analysis.to_string(index=False))
        grade_analysis.to_csv(output_path / "per_grade_analysis.csv", index=False)
    
    # --- Save best model ---
    model_path = output_path / "grade_predictor.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": best_model,
            "model_name": best_name,
            "feature_columns": FEATURE_COLUMNS,
            "metrics": all_metrics,
        }, f)
    print(f"\nSaved best model ({best_name}) to {model_path}")
    
    # Save metrics
    with open(output_path / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
