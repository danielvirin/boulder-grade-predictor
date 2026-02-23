"""
Feature Engineering: Extract predictive features from climb hold configurations.

Takes the raw hold positions and roles for each climb and computes features
that capture the physical difficulty of the climb:

1. Spatial features — distances, spans, heights
2. Hold composition — counts and ratios by role
3. Movement complexity — direction changes, cross-body moves
4. Board angle — normalised wall angle

The key insight is that harder climbs tend to have:
- Longer moves between holds
- Fewer footholds relative to handholds
- More lateral movement (cross-body moves)
- Higher average hold positions
- Steeper board angles
"""

import pandas as pd
import numpy as np
from pathlib import Path

from data_pipeline import ROLE_START, ROLE_HAND, ROLE_FINISH, ROLE_FOOT


def compute_spatial_features(holds: pd.DataFrame) -> dict:
    """
    Compute spatial features from a climb's hold positions.

    Sorts holds bottom-to-top (by y coordinate) to approximate the
    climbing sequence, then computes distances and spans.
    """
    if len(holds) < 2:
        return {
            "total_distance": 0,
            "max_move_distance": 0,
            "avg_move_distance": 0,
            "vertical_span": 0,
            "lateral_span": 0,
            "avg_hold_height": holds["y"].mean() if len(holds) > 0 else 0,
            "max_hold_height": holds["y"].max() if len(holds) > 0 else 0,
            "min_hold_height": holds["y"].min() if len(holds) > 0 else 0,
        }

    # Sort by y position (bottom to top) as a proxy for climbing order
    sorted_holds = holds.sort_values(["y", "x"]).reset_index(drop=True)

    # Compute distances between consecutive holds
    dx = sorted_holds["x"].diff().dropna()
    dy = sorted_holds["y"].diff().dropna()
    distances = np.sqrt(dx**2 + dy**2)

    return {
        "total_distance": distances.sum(),
        "max_move_distance": distances.max(),
        "avg_move_distance": distances.mean(),
        "vertical_span": holds["y"].max() - holds["y"].min(),
        "lateral_span": holds["x"].max() - holds["x"].min(),
        "avg_hold_height": holds["y"].mean(),
        "max_hold_height": holds["y"].max(),
        "min_hold_height": holds["y"].min(),
    }


def compute_hold_composition(holds: pd.DataFrame) -> dict:
    """
    Compute hold count and composition features.

    Harder climbs tend to have fewer footholds relative to handholds,
    forcing climbers to use handholds as footholds (more demanding).
    """
    role_counts = holds["role"].value_counts().to_dict()

    n_start = role_counts.get(ROLE_START, 0)
    n_hand = role_counts.get(ROLE_HAND, 0)
    n_finish = role_counts.get(ROLE_FINISH, 0)
    n_foot = role_counts.get(ROLE_FOOT, 0)
    n_total = len(holds)

    foot_ratio = n_foot / n_total if n_total > 0 else 0
    n_hand_moves = n_hand + n_finish + n_start

    return {
        "n_holds_total": n_total,
        "n_start": n_start,
        "n_hand": n_hand,
        "n_finish": n_finish,
        "n_foot": n_foot,
        "n_hand_moves": n_hand_moves,
        "foot_ratio": foot_ratio,
    }


def compute_movement_features(holds: pd.DataFrame) -> dict:
    """
    Compute movement complexity features.

    Analyses the climbing sequence for direction changes, cross-body moves,
    and other patterns that increase difficulty.
    """
    if len(holds) < 3:
        return {
            "n_direction_changes": 0,
            "max_lateral_move": 0,
            "avg_lateral_move": 0,
            "max_vertical_gap": 0,
            "has_big_cross_body": 0,
            "compactness": 0,
        }

    sorted_holds = holds.sort_values(["y", "x"]).reset_index(drop=True)

    lateral_moves = sorted_holds["x"].diff().dropna()
    vertical_moves = sorted_holds["y"].diff().dropna()

    # Direction changes (sign changes in lateral movement)
    signs = np.sign(lateral_moves)
    sign_changes = (signs.diff().dropna() != 0).sum()

    # Cross-body detection
    board_width = holds["x"].max() - holds["x"].min()
    cross_body_threshold = board_width * 0.4 if board_width > 0 else float("inf")
    has_big_cross_body = int(lateral_moves.abs().max() > cross_body_threshold)

    # Compactness: average distance from centroid
    centroid_x = holds["x"].mean()
    centroid_y = holds["y"].mean()
    distances_from_centroid = np.sqrt(
        (holds["x"] - centroid_x) ** 2 + (holds["y"] - centroid_y) ** 2
    )
    compactness = distances_from_centroid.mean()

    return {
        "n_direction_changes": int(sign_changes),
        "max_lateral_move": lateral_moves.abs().max(),
        "avg_lateral_move": lateral_moves.abs().mean(),
        "max_vertical_gap": vertical_moves.abs().max(),
        "has_big_cross_body": has_big_cross_body,
        "compactness": compactness,
    }


def compute_hand_sequence_features(holds: pd.DataFrame) -> dict:
    """
    Compute features based on hand holds only (excluding feet).

    This better approximates the actual movement sequence since
    footholds are often used statically.
    """
    hand_holds = holds[holds["role"].isin([ROLE_START, ROLE_HAND, ROLE_FINISH])]

    if len(hand_holds) < 2:
        return {
            "hand_total_distance": 0,
            "hand_max_move": 0,
            "hand_avg_move": 0,
            "hand_vertical_span": 0,
        }

    sorted_hands = hand_holds.sort_values(["y", "x"]).reset_index(drop=True)
    dx = sorted_hands["x"].diff().dropna()
    dy = sorted_hands["y"].diff().dropna()
    distances = np.sqrt(dx**2 + dy**2)

    return {
        "hand_total_distance": distances.sum(),
        "hand_max_move": distances.max(),
        "hand_avg_move": distances.mean(),
        "hand_vertical_span": hand_holds["y"].max() - hand_holds["y"].min(),
    }


def engineer_features(
    climbs_df: pd.DataFrame,
    holds_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Engineer all features for each climb.

    Combines spatial, hold composition, movement, and hand sequence features
    with the board angle to produce the final feature matrix.
    """
    feature_records = []

    climb_ids = climbs_df["climb_id"].unique()
    total = len(climb_ids)

    for i, climb_id in enumerate(climb_ids):
        if (i + 1) % 500 == 0:
            print(f"  Processing climb {i + 1}/{total}...")

        climb_holds = holds_df[holds_df["climb_id"] == climb_id]
        climb_info = climbs_df[climbs_df["climb_id"] == climb_id].iloc[0]

        if len(climb_holds) == 0:
            continue

        features = {"climb_id": climb_id}

        # Board angle
        features["angle"] = climb_info["angle"]
        features["angle_normalised"] = climb_info["angle"] / 70.0

        # All feature groups
        features.update(compute_spatial_features(climb_holds))
        features.update(compute_hold_composition(climb_holds))
        features.update(compute_movement_features(climb_holds))
        features.update(compute_hand_sequence_features(climb_holds))

        # Targets
        features["target_grade"] = climb_info["community_grade"]
        features["setter_grade"] = climb_info["setter_grade"]
        if "community_v_num" in climb_info:
            features["target_v_num"] = climb_info["community_v_num"]
            features["setter_v_num"] = climb_info["setter_v_num"]

        feature_records.append(features)

    features_df = pd.DataFrame(feature_records)
    print(f"\n  Engineered features for {len(features_df)} climbs")
    print(f"  Feature columns: {len(features_df.columns) - 5}")

    return features_df


def get_feature_columns() -> list[str]:
    """Return the list of feature column names used for training."""
    return [
        "angle_normalised",
        "total_distance",
        "max_move_distance",
        "avg_move_distance",
        "vertical_span",
        "lateral_span",
        "avg_hold_height",
        "max_hold_height",
        "min_hold_height",
        "n_holds_total",
        "n_start",
        "n_hand",
        "n_finish",
        "n_foot",
        "n_hand_moves",
        "foot_ratio",
        "n_direction_changes",
        "max_lateral_move",
        "avg_lateral_move",
        "max_vertical_gap",
        "has_big_cross_body",
        "compactness",
        "hand_total_distance",
        "hand_max_move",
        "hand_avg_move",
        "hand_vertical_span",
    ]


def run_feature_engineering(data_dir: str = "data"):
    """Run feature engineering on processed data."""
    print("=" * 60)
    print("Boulder Grade Predictor — Feature Engineering")
    print("=" * 60)

    data_path = Path(data_dir)

    print("\n1. Loading processed data...")
    climbs_df = pd.read_csv(data_path / "climbs_processed.csv")
    holds_df = pd.read_csv(data_path / "climb_holds.csv")
    print(f"   {len(climbs_df)} climbs, {len(holds_df)} holds")

    print("\n2. Engineering features...")
    features_df = engineer_features(climbs_df, holds_df)

    output_path = data_path / "features.csv"
    features_df.to_csv(output_path, index=False)
    print(f"\n✅ Features saved to {output_path}")

    # Feature summary
    print("\n📊 Feature summary:")
    feature_cols = get_feature_columns()
    for col in feature_cols:
        if col in features_df.columns:
            print(
                f"   {col:30s} "
                f"mean={features_df[col].mean():8.2f}  "
                f"std={features_df[col].std():8.2f}"
            )

    return features_df


if __name__ == "__main__":
    run_feature_engineering()
