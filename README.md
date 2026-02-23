# 🧗 Boulder Grade Predictor

**Predicting Kilter Board boulder problem difficulty using machine learning.**

Climbing grades are subjective — the same problem can feel V5 to one climber and V7 to another. This project uses the Kilter Board's rich dataset of climbs, hold configurations, and community consensus grades to build a model that predicts difficulty from a climb's physical characteristics.

## The Problem

The Kilter Board app lets anyone set a climb and assign a grade. Other climbers can then log ascents and vote on whether the grade feels accurate. But the app's "Quick Log Ascent" feature auto-logs a flash at the setter's grade, flooding the grade histogram with potentially inaccurate data. This makes it hard to find climbs that are truly at a given grade.

**This model cuts through the noise** — it predicts what grade a climb *should* be based on its hold configuration, spatial properties, and board angle, independent of the setter's opinion.

## Features

- Extracts and processes climb data from the Kilter Board SQLite database
- Engineers spatial and biomechanical features from hold positions and types
- Trains gradient-boosted models (XGBoost) to predict V-grade from climb characteristics
- Evaluates accuracy within ±1 grade (the standard for consensus in climbing)
- Identifies "sandbagged" and "soft" climbs where predicted grade diverges from the set grade
- Interactive analysis notebook with visualisations

## Project Structure

```
boulder-grade-predictor/
├── README.md
├── requirements.txt
├── setup_data.sh              # Script to download Kilter Board database
├── data/                      # Raw and processed data (gitignored)
├── src/
│   ├── data_pipeline.py       # Extract and process data from Kilter DB
│   ├── feature_engineering.py # Spatial and hold-based feature extraction
│   ├── train.py               # Model training and evaluation
│   └── predict.py             # Grade prediction for new climbs
├── notebooks/
│   └── analysis.ipynb         # EDA and results visualisation
└── models/                    # Saved trained models (gitignored)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Kilter Board database

```bash
bash setup_data.sh
```

This uses [BoardLib](https://github.com/lemeryfertitta/BoardLib) to download the official Kilter Board SQLite database.

### 3. Process data and train

```bash
python src/data_pipeline.py      # Extract climbs, holds, and ascents
python src/feature_engineering.py # Generate features from hold configurations
python src/train.py               # Train the model and evaluate
```

### 4. Predict a grade

```bash
python src/predict.py --climb-id 12345
```

## Methodology

### Data Source

The Kilter Board app stores all climb and ascent data in a SQLite database accessible via the Aurora Climbing API. Each climb consists of:
- A set of holds with (x, y) positions on the board grid
- Hold roles: start (green), hand (cyan), foot (orange), finish (purple)
- The board angle (0°–70°)
- Community ascent logs with user-submitted grade opinions

### Feature Engineering

From each climb's hold configuration, we extract:

**Spatial features:**
- Total distance (sum of consecutive hold-to-hold distances)
- Max single move distance
- Average move distance
- Vertical span (highest hold - lowest hold)
- Lateral span (widest horizontal distance)
- Average hold height on the board

**Hold composition:**
- Number of hand holds, foot holds, start holds, finish holds
- Total hold count
- Ratio of footholds to handholds

**Movement complexity:**
- Number of direction changes (lateral reversals)
- Max vertical gap between consecutive holds
- Presence of cross-body moves (large lateral shifts)

**Board angle** (normalised 0–1)

### Model

We use XGBoost regression to predict the continuous Kilter Board internal grade scale (which maps to V-grades). XGBoost was chosen for:
- Strong performance on tabular data with engineered features
- Built-in feature importance for interpretability
- Fast training iteration

### Evaluation

- **Primary metric:** Percentage of predictions within ±1 V-grade of community consensus
- **Secondary:** MAE (Mean Absolute Error) in V-grade units
- **Sanity check:** Feature importance analysis to verify the model learns sensible relationships (e.g., longer moves and steeper angles → harder grades)

## Results

*Run the pipeline with real data to generate results. See `notebooks/analysis.ipynb` for full analysis.*

## Data Attribution

Climb data is sourced from the [Kilter Board](https://kilterboard.com/) via [BoardLib](https://github.com/lemeryfertitta/BoardLib). This project is for educational and research purposes.

## License

MIT
