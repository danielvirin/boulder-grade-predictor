"""
data_processing.py — Parse Kilter Board SQLite database into clean DataFrames.

Handles both:
  - Raw BoardLib SQLite databases (downloaded via `boardlib database kilter ...`)
  - The pre-cleaned HuggingFace dataset (Vilin97/KilterBoard)

Layout encoding: each climb's holds are stored as a string like
  "p1083r15p1117r15p1164r12p1185r12p1233r13"
where pXXXX = placement_id and rYY = role_id:
  r12 = start (green), r13 = hand (blue), r14 = finish (purple), r15 = foot (orange)
"""

import argparse
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


# --- Constants ---

ROLE_MAP = {
    12: "start",
    13: "hand",
    14: "finish",
    15: "foot",
}

# Kilter Board internal grade scale → V-grade mapping
# Internal scale runs 0–40, roughly corresponding to font grades 1a–9c
# We map the commonly used range to V-grades
GRADE_MAP = {
    10: "V0", 11: "V0", 12: "V1", 13: "V1",
    14: "V2", 15: "V2", 16: "V3", 17: "V3",
    18: "V4", 19: "V4", 20: "V5", 21: "V5",
    22: "V6", 23: "V6", 24: "V7", 25: "V7",
    26: "V8", 27: "V8", 28: "V9", 29: "V9",
    30: "V10", 31: "V10", 32: "V11", 33: "V11",
    34: "V12", 35: "V12", 36: "V13", 37: "V13",
    38: "V14", 39: "V14", 40: "V15",
}


def parse_layout_string(layout: str) -> list[dict]:
    """Parse a climb layout string into a list of hold dicts.
    
    Args:
        layout: String like "p1083r15p1117r15p1164r12"
        
    Returns:
        List of dicts with keys: placement_id, role_id, role_name
    """
    pattern = r"p(\d+)r(\d+)"
    matches = re.findall(pattern, layout)
    
    holds = []
    for placement_id, role_id in matches:
        role_id = int(role_id)
        holds.append({
            "placement_id": int(placement_id),
            "role_id": role_id,
            "role_name": ROLE_MAP.get(role_id, f"unknown_{role_id}"),
        })
    
    return holds


def load_raw_database(db_path: str) -> dict[str, pd.DataFrame]:
    """Load tables from a raw BoardLib SQLite database.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        Dict of table name → DataFrame
    """
    conn = sqlite3.connect(db_path)
    
    tables = {}
    
    # Core climb data
    tables["climbs"] = pd.read_sql_query("""
        SELECT 
            uuid,
            name,
            description,
            frames AS layout,
            setter_username,
            is_listed,
            layout_id,
            edge_left,
            edge_right,
            edge_bottom,
            edge_top
        FROM climbs
        WHERE is_listed = 1
    """, conn)
    
    # Climb stats (ascents, quality, grade votes)
    tables["climb_stats"] = pd.read_sql_query("""
        SELECT 
            climb_uuid,
            angle,
            ascensionist_count,
            difficulty_average,
            quality_average,
            benchmark_difficulty
        FROM climb_stats
    """, conn)
    
    # Placements (hold_id → hole_id mapping)
    tables["placements"] = pd.read_sql_query("""
        SELECT 
            id AS placement_id,
            hole_id,
            layout_id,
            default_placement_role_id
        FROM placements
    """, conn)
    
    # Holes (x, y coordinates)
    tables["holes"] = pd.read_sql_query("""
        SELECT 
            id AS hole_id,
            x,
            y
        FROM holes
    """, conn)
    
    # Difficulty grades lookup
    tables["difficulty_grades"] = pd.read_sql_query("""
        SELECT 
            difficulty,
            boulder_name,
            route_name
        FROM difficulty_grades
    """, conn)
    
    conn.close()
    
    return tables


def load_huggingface_database(db_path: str) -> dict[str, pd.DataFrame]:
    """Load tables from the HuggingFace pre-cleaned dataset.
    
    The HF dataset has tables: kilter_train, kilter_val, kilter_test,
    plus reference tables: difficulty_grades, placements, placement_roles, holes
    
    Args:
        db_path: Path to kilter_splits.sqlite
        
    Returns:
        Dict of table name → DataFrame
    """
    conn = sqlite3.connect(db_path)
    
    tables = {}
    
    # Get list of all tables
    table_list = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", conn
    )["name"].tolist()
    
    for table_name in table_list:
        tables[table_name] = pd.read_sql_query(
            f"SELECT * FROM {table_name}", conn
        )
    
    conn.close()
    
    return tables


def build_climb_dataset(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a unified climb dataset from raw database tables.
    
    Joins climbs with stats, parses layout strings, and resolves hold
    positions from placements → holes.
    
    Args:
        tables: Dict of table DataFrames from load_raw_database()
        
    Returns:
        DataFrame with one row per (climb, angle) combination, including
        parsed hold positions and metadata
    """
    climbs = tables["climbs"]
    stats = tables["climb_stats"]
    placements = tables["placements"]
    holes = tables["holes"]
    
    # Join climbs with stats
    df = climbs.merge(
        stats,
        left_on="uuid",
        right_on="climb_uuid",
        how="inner",
    )
    
    # Filter: need at least some ascents for a reliable grade
    df = df[df["ascensionist_count"] > 0].copy()
    
    # Build placement → (x, y) lookup
    placement_coords = placements.merge(holes, on="hole_id", how="left")
    coord_lookup = dict(
        zip(
            placement_coords["placement_id"],
            zip(placement_coords["x"], placement_coords["y"]),
        )
    )
    
    # Parse layouts and resolve coordinates
    parsed_rows = []
    
    for _, row in df.iterrows():
        layout = row["layout"]
        if not layout or not isinstance(layout, str):
            continue
            
        holds = parse_layout_string(layout)
        if len(holds) < 2:
            continue
        
        # Resolve coordinates for each hold
        resolved_holds = []
        for hold in holds:
            coords = coord_lookup.get(hold["placement_id"])
            if coords:
                resolved_holds.append({
                    **hold,
                    "x": coords[0],
                    "y": coords[1],
                })
        
        if len(resolved_holds) < 2:
            continue
        
        parsed_rows.append({
            "uuid": row["uuid"],
            "name": row["name"],
            "angle": row["angle"],
            "layout": layout,
            "setter_username": row.get("setter_username", ""),
            "ascensionist_count": row["ascensionist_count"],
            "difficulty_average": row["difficulty_average"],
            "quality_average": row.get("quality_average", np.nan),
            "holds": resolved_holds,
            "num_holds": len(resolved_holds),
        })
    
    result = pd.DataFrame(parsed_rows)
    
    # Map internal difficulty to V-grade
    result["v_grade_numeric"] = result["difficulty_average"].round().astype(int)
    result["v_grade"] = result["v_grade_numeric"].map(GRADE_MAP)
    
    return result


def build_climb_dataset_from_hf(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a unified dataset from the HuggingFace pre-cleaned data.
    
    Args:
        tables: Dict of table DataFrames from load_huggingface_database()
        
    Returns:
        DataFrame with parsed hold positions, with a 'split' column
    """
    placements = tables.get("placements", pd.DataFrame())
    holes = tables.get("holes", pd.DataFrame())
    
    # Build coordinate lookup
    if not placements.empty and not holes.empty:
        placement_coords = placements.merge(holes, on="hole_id", how="left")
        coord_lookup = dict(
            zip(
                placement_coords["placement_id"] if "placement_id" in placement_coords.columns
                else placement_coords["id"],
                zip(placement_coords["x"], placement_coords["y"]),
            )
        )
    else:
        coord_lookup = {}
    
    all_rows = []
    
    for split_name in ["kilter_train", "kilter_val", "kilter_test"]:
        if split_name not in tables:
            continue
            
        split_df = tables[split_name]
        split_label = split_name.replace("kilter_", "")
        
        for _, row in split_df.iterrows():
            # The HF dataset may use 'frames' or 'layout' for the hold string
            layout = row.get("frames", row.get("layout", ""))
            if not layout or not isinstance(layout, str):
                continue
            
            holds = parse_layout_string(layout)
            if len(holds) < 2:
                continue
            
            resolved_holds = []
            for hold in holds:
                coords = coord_lookup.get(hold["placement_id"])
                if coords:
                    resolved_holds.append({**hold, "x": coords[0], "y": coords[1]})
            
            if len(resolved_holds) < 2:
                continue
            
            all_rows.append({
                "uuid": row.get("uuid", ""),
                "name": row.get("name", ""),
                "angle": row.get("angle", 0),
                "layout": layout,
                "ascensionist_count": row.get("ascensionist_count", 0),
                "difficulty_average": row.get("difficulty_average", 0),
                "quality_average": row.get("quality_average", np.nan),
                "holds": resolved_holds,
                "num_holds": len(resolved_holds),
                "split": split_label,
            })
    
    result = pd.DataFrame(all_rows)
    
    if not result.empty:
        result["v_grade_numeric"] = result["difficulty_average"].round().astype(int)
        result["v_grade"] = result["v_grade_numeric"].map(GRADE_MAP)
    
    return result


def save_processed_data(df: pd.DataFrame, output_dir: str) -> None:
    """Save processed climb data to parquet and CSV.
    
    Args:
        df: Processed climb DataFrame
        output_dir: Directory to save files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full dataset (without the holds list column for parquet compat)
    df_save = df.drop(columns=["holds"])
    df_save.to_parquet(output_path / "climbs_processed.parquet", index=False)
    df_save.to_csv(output_path / "climbs_processed.csv", index=False)
    
    # Save holds as separate file (one row per hold per climb)
    holds_rows = []
    for _, row in df.iterrows():
        for i, hold in enumerate(row["holds"]):
            holds_rows.append({
                "uuid": row["uuid"],
                "angle": row["angle"],
                "hold_index": i,
                "placement_id": hold["placement_id"],
                "role_id": hold["role_id"],
                "role_name": hold["role_name"],
                "x": hold["x"],
                "y": hold["y"],
            })
    
    holds_df = pd.DataFrame(holds_rows)
    holds_df.to_parquet(output_path / "holds.parquet", index=False)
    holds_df.to_csv(output_path / "holds.csv", index=False)
    
    print(f"Saved {len(df)} climbs to {output_path}")
    print(f"Saved {len(holds_df)} hold records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process Kilter Board database")
    parser.add_argument(
        "--db-path", type=str, required=True,
        help="Path to SQLite database (BoardLib or HuggingFace)"
    )
    parser.add_argument(
        "--output", type=str, default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--source", type=str, choices=["boardlib", "huggingface"], default="boardlib",
        help="Database source format"
    )
    parser.add_argument(
        "--min-ascents", type=int, default=10,
        help="Minimum ascensionist count to include a climb"
    )
    args = parser.parse_args()
    
    print(f"Loading database from {args.db_path} (source: {args.source})...")
    
    if args.source == "huggingface":
        tables = load_huggingface_database(args.db_path)
        df = build_climb_dataset_from_hf(tables)
    else:
        tables = load_raw_database(args.db_path)
        df = build_climb_dataset(tables)
    
    print(f"Loaded {len(df)} climb-angle combinations")
    
    # Filter by minimum ascents
    df = df[df["ascensionist_count"] >= args.min_ascents].copy()
    print(f"After filtering (>= {args.min_ascents} ascents): {len(df)} climbs")
    
    # Summary stats
    print(f"\nGrade distribution:")
    print(df["v_grade"].value_counts().sort_index())
    print(f"\nAngle distribution:")
    print(df["angle"].value_counts().sort_index())
    
    save_processed_data(df, args.output)


if __name__ == "__main__":
    main()
