"""
Predict: Grade prediction for individual Kilter Board climbs.

Takes a climb ID (from the Kilter Board app) or a hold configuration
and returns a predicted grade with confidence information.
"""

import argparse
import pickle
import sqlite3
import re
import numpy as np
import pandas as pd
from pathlib import Path

from data_pipeline import GRADE_TO_V, LAYOUT_ID, connect_db
from feature_engineering import (
    compute_spatial_features,
    compute_hold_composition,
    compute_movement_features,
    compute_hand_sequence_features,
    get_feature_columns,
)


def load_model(model_dir: str = "models"):
    """Load the trained model."""
    model_path = Path(model_dir) / "grade_predictor.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py first."
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def internal_to_v_grade(internal_grade: float) -> str:
    """Convert internal grade scale to V-grade string."""
    rounded = int(round(internal_grade))
    rounded = max(10, min(40, rounded))
    return GRADE_TO_V.get(rounded, f"~V{(rounded - 10) // 2}")


def get_climb_data(conn: sqlite3.Connection, climb_id: str) -> tuple:
    """
    Fetch a climb's hold configuration and metadata from the database.

    Returns (climb_info_dict, holds_dataframe).
    """
    # Get climb metadata
    climb_query = """
    SELECT
        c.uuid AS climb_id,
        c.name,
        c.frames,
        c.difficulty AS setter_grade,
        c.angle,
        cs.display_difficulty AS community_grade,
        cs.ascensionist_count AS ascents
    FROM climbs c
    LEFT JOIN climb_stats cs ON c.uuid = cs.climb_uuid
    WHERE c.uuid = ?
    """
    climb_df = pd.read_sql_query(climb_query, conn, params=(climb_id,))

    if climb_df.empty:
        raise ValueError(f"Climb '{climb_id}' not found in database.")

    climb_info = climb_df.iloc[0].to_dict()

    # Parse hold configuration
    frames = climb_info["frames"]
    pattern = r'p(\d+)r(\d+)'
    matches = re.findall(pattern, frames or "")

    hold_records = []
    for placement_id, role_id in matches:
        hold_records.append({
            "placement_id": int(placement_id),
            "role": int(role_id),
        })

    holds_df = pd.DataFrame(hold_records)

    # Get physical positions
    if not holds_df.empty:
        placement_ids = holds_df["placement_id"].tolist()
        placeholders = ",".join(["?"] * len(placement_ids))
        pos_query = f"""
        SELECT p.id AS placement_id, h.x, h.y
        FROM placements p
        JOIN holes h ON p.hole_id = h.id
        WHERE p.id IN ({placeholders})
        """
        positions_df = pd.read_sql_query(pos_query, conn, params=placement_ids)
        holds_df = holds_df.merge(positions_df, on="placement_id", how="left")

    return climb_info, holds_df


def predict_grade(
    holds_df: pd.DataFrame,
    angle: float,
    model,
) -> dict:
    """
    Predict the grade for a climb given its holds and angle.

    Returns a dict with predicted grade and feature values.
    """
    features = {}

    # Board angle
    features["angle_normalised"] = angle / 70.0

    # Compute all feature groups
    features.update(compute_spatial_features(holds_df))
    features.update(compute_hold_composition(holds_df))
    features.update(compute_movement_features(holds_df))
    features.update(compute_hand_sequence_features(holds_df))

    # Build feature vector in correct order
    feature_cols = get_feature_columns()
    X = np.array([[features[col] for col in feature_cols]])

    # Predict
    pred_internal = model.predict(X)[0]
    pred_v = internal_to_v_grade(pred_internal)

    return {
        "predicted_internal": float(pred_internal),
        "predicted_v_grade": pred_v,
        "features": features,
    }


def predict_from_climb_id(
    climb_id: str,
    db_path: str = "data/kilter.sqlite",
    model_dir: str = "models",
) -> dict:
    """
    Predict the grade for a climb by its ID in the Kilter Board database.

    Also compares with the setter's grade and community consensus.
    """
    model = load_model(model_dir)
    conn = connect_db(db_path)

    climb_info, holds_df = get_climb_data(conn, climb_id)
    conn.close()

    result = predict_grade(holds_df, climb_info["angle"], model)

    # Add comparison info
    setter_v = internal_to_v_grade(climb_info["setter_grade"]) if climb_info["setter_grade"] else "Unknown"
    community_v = internal_to_v_grade(climb_info["community_grade"]) if climb_info.get("community_grade") else "Unknown"

    result["climb_name"] = climb_info["name"]
    result["angle"] = climb_info["angle"]
    result["setter_grade"] = setter_v
    result["community_grade"] = community_v
    result["ascents"] = climb_info.get("ascents", 0)

    return result


def format_prediction(result: dict) -> str:
    """Format a prediction result for display."""
    lines = [
        "",
        f"🧗 {result.get('climb_name', 'Unknown Climb')}",
        f"   Angle: {result.get('angle', '?')}°",
        f"   Ascents: {result.get('ascents', '?')}",
        "",
        f"   Setter grade:     {result.get('setter_grade', '?')}",
        f"   Community grade:  {result.get('community_grade', '?')}",
        f"   Predicted grade:  {result['predicted_v_grade']}",
        "",
    ]

    # Flag disagreements
    pred = result["predicted_v_grade"]
    setter = result.get("setter_grade", "")
    if pred != setter and setter != "Unknown":
        if pred > setter:
            lines.append(f"   ⚠️  Potentially sandbagged (harder than set grade)")
        else:
            lines.append(f"   ⚠️  Potentially soft (easier than set grade)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Predict Kilter Board climb grade")
    parser.add_argument("--climb-id", required=True, help="Climb UUID from Kilter Board database")
    parser.add_argument("--db", default="data/kilter.sqlite", help="Path to Kilter Board database")
    parser.add_argument("--model", default="models", help="Path to model directory")

    args = parser.parse_args()

    result = predict_from_climb_id(args.climb_id, args.db, args.model)
    print(format_prediction(result))


if __name__ == "__main__":
    main()
