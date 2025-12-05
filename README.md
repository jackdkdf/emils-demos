# CS2 Match Prediction ML Project

A machine learning project for predicting CS2 match outcomes using team statistics, player performance, and historical data.

## Project Structure

```
emils-demos/
├── src/                    # Main source code package
│   ├── data/              # Data preprocessing modules
│   │   ├── preprocess.py  # Main preprocessing pipeline
│   │   ├── team_mapping.py
│   │   ├── map_mapping.py
│   │   ├── cumulative_stats.py
│   │   ├── match_features.py
│   │   ├── final_features.py
│   │   ├── player_stats.py
│   │   └── utils.py
│   ├── run/               # Training entrypoints
│   │   └── train.py      # Model training scripts
│   ├── eval/              # Evaluation scripts and metrics
│   │   └── metrics.py
│   └── main.py           # Main CLI entrypoint
├── notebooks/             # Jupyter notebooks for exploration
│   ├── preprocess.ipynb
│   ├── model.ipynb
│   └── xgboost.ipynb
├── data/                  # Data directory (organized by type)
│   ├── raw/              # Raw/unprocessed data
│   │   ├── team_results/      # Team match results CSV files
│   │   ├── player_results/   # Player weekly statistics CSV files
│   │   └── rankings/          # Ranking files
│   │       ├── hltv_team_rankings_original.csv
│   │       └── teams_peak_36.csv
│   ├── preprocessed/     # Processed/preprocessed data
│   │   ├── final_features.csv
│   │   ├── match_features.csv
│   │   ├── team_map_cumulative_stats.csv
│   │   └── team_opponent_cumulative_stats.csv
│   └── mappings/         # Mapping/metadata files
│       ├── team_name_to_id.csv
│       └── map_name_to_id.csv
├── scripts/              # One-off utility scripts
├── tests/                # Unit and integration tests
├── plots/                # Generated visualization plots
└── pyproject.toml        # Poetry configuration

```

## Features

The model uses the following features to predict match outcomes:

- **Team vs Team Statistics**: Cumulative wins/losses and win rate between two teams
- **Map Performance**: Each team's win rate and record on specific maps
- **Global Rankings**: HLTV ranking points for both teams at match time
- **Win/Loss Streaks**: Current winning and losing streaks for both teams
- **Player Statistics**: Overall rating, utility success, and opening rating for top 5 players per team (30 features total)
- **Map One-Hot Encoding**: One-hot encoded map IDs

## Installation

### Using Poetry (Recommended)

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Using pip

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

## Usage

### Command Line Interface

The project provides a CLI for common tasks:

```bash
# Run full preprocessing pipeline
python -m src.main preprocess

# Train XGBoost model (includes evaluation metrics)
python -m src.main train --data data/preprocessed/final_features.csv

# Create visualizations (confusion matrix, ROC curve, feature importance)
python -m src.main visualize --data data/preprocessed/final_features.csv

# Show project info
python -m src.main about
```

#### CLI Options

**Preprocess Command:**

```bash
python -m src.main preprocess [--project-root PATH] [--quiet]
```

**Train Command:**

```bash
python -m src.main train --data PATH [--no-balance] [--no-tuning] [--n-iter N] [--seed N] [--quiet]
```

**Visualize Command:**

```bash
python -m src.main visualize --data PATH [--output-dir DIR] [--no-train] [--model-path PATH] [--seed N] [--quiet] [--show]
```

The visualize command creates:

- Class distribution bar charts (before and after balancing)
- Confusion matrix heatmap
- ROC curve
- Precision-Recall curve
- Feature importance plot
- Loss curves (training and validation log loss over boosting rounds)
- Accuracy curves (training and validation accuracy over boosting rounds) (top 20 features)

### Full Workflow Example

Here's a complete workflow to preprocess data and train a model:

```bash
# Step 1: Run preprocessing pipeline
# This creates all necessary intermediate files and final_features.csv
# Output: Creates files in data/preprocessed/ and data/mappings/
python -m src.main preprocess

# Step 2: Train the model with hyperparameter tuning
# This performs RandomizedSearchCV to find best hyperparameters, then trains final model
# Output: Trained model and evaluation metrics on test set
python -m src.main train --data data/preprocessed/final_features.csv

# Step 3: Create visualizations
# This generates plots for confusion matrix, ROC curve, precision-recall curve, and feature importance
# Output: PNG files saved to plots/ directory
python -m src.main visualize --data data/preprocessed/final_features.csv
```

**Note**: The `train` command automatically evaluates the model on the test set and displays metrics after training. The `visualize` command can train a model or use an existing one to generate visualizations.

### Testing the Full Workflow

To test the complete pipeline from scratch:

```bash
# 1. Verify data structure exists
ls -la data/raw/team_results/      # Should show team CSV files
ls -la data/raw/player_results/   # Should show player stats CSV files
ls -la data/raw/rankings/         # Should show ranking CSV files

# 2. Run preprocessing (this may take a few minutes)
python -m src.main preprocess

# Verify preprocessing outputs
ls -la data/preprocessed/         # Should show processed CSV files
ls -la data/mappings/             # Should show mapping CSV files

# 3. Train model (this may take several minutes due to hyperparameter tuning)
#    The train command automatically evaluates and displays metrics
python -m src.main train --data data/preprocessed/final_features.csv

# 4. (Optional) Train without hyperparameter tuning for faster testing
python -m src.main train --data data/preprocessed/final_features.csv --no-tuning

# 6. (Optional) Train without class balancing
python -m src.main train --data data/preprocessed/final_features.csv --no-balance
```

### Python API

You can also use the modules directly in Python:

```python
from pathlib import Path
from src.data.preprocess import run_preprocessing_pipeline
from src.run.train import load_and_prepare_data, train_xgboost_model
from src.eval.metrics import evaluate_model

# Run preprocessing
run_preprocessing_pipeline()

# Load data and train model
X_train, X_test, y_train, y_test = load_and_prepare_data(
    data_file=Path("data/preprocessed/final_features.csv"),
    balance_classes=True
)

model, best_params = train_xgboost_model(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    hyperparameter_tuning=True
)

# Evaluate
y_pred = model.predict(X_test)
evaluate_model(y_test.values, y_pred)
```

### Jupyter Notebooks

The notebooks in the `notebooks/` directory provide interactive exploration:

- `preprocess.ipynb`: Data preprocessing and feature engineering
- `model.ipynb`: Neural network model training
- `xgboost.ipynb`: XGBoost model training with hyperparameter tuning

## Data Preprocessing Pipeline

The preprocessing pipeline consists of 5 main steps:

1. **Team Mapping**: Create team name to ID mapping from `data/raw/rankings/teams_peak_36.csv`
2. **Map Mapping**: Extract map name to ID mapping from `data/raw/team_results/`
3. **Opponent Statistics**: Calculate cumulative wins/losses for each team against each opponent
4. **Map Statistics**: Calculate cumulative wins/losses for each team on each map
5. **Match Features**: Combine all statistics with rankings, streaks, and player stats to create feature dataset

All intermediate files are saved to `data/preprocessed/` and mapping files to `data/mappings/`.

## Model

The project uses **XGBoost** (Gradient Boosting) for match prediction:

- **Objective**: Binary classification (team A wins vs loses)
- **Hyperparameter Tuning**: RandomizedSearchCV with 50 random combinations
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Validation**: 80/20 train/test split for model evaluation
- **Class Balancing**: Optional undersampling to balance classes (enabled by default)

## Logging

The project uses Python's `logging` module for all output. Log levels:

- **INFO**: Default level, shows progress and results
- **WARNING**: Warnings and non-critical issues
- **ERROR**: Errors during processing
- **DEBUG**: Detailed debugging information (use `--quiet` flag to suppress)

Log format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

The project follows PEP 8 style guidelines. Consider using `black` for code formatting:

```bash
pip install black
black src/
```

## Data Organization

The `data/` folder is organized into three main categories:

- **`raw/`**: Original, unprocessed data files

  - `team_results/`: Individual team match CSV files
  - `player_results/`: Player weekly statistics CSV files
  - `rankings/`: Team ranking files

- **`preprocessed/`**: Processed and intermediate data files

  - Cumulative statistics
  - Match features
  - Final feature sets ready for training

- **`mappings/`**: Metadata and mapping files
  - Team name to ID mappings
  - Map name to ID mappings

## Output Files

The `plots/` directory contains visualization outputs generated by the `visualize` command:

- **`class_distribution.png`**: Bar charts showing class distribution before and after balancing
- **`confusion_matrix.png`**: Confusion matrix heatmap showing model predictions
- **`roc_curve.png`**: ROC curve with AUC score
- **`precision_recall_curve.png`**: Precision-Recall curve
- **`feature_importance.png`**: Top 20 most important features
- **`loss_curves.png`**: Training and validation loss curves over boosting rounds
- **`accuracy_curves.png`**: Training and validation accuracy curves over boosting rounds

## License

See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- Data sources: HLTV team rankings and match results
- Player statistics from weekly performance data
