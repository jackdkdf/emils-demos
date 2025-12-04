"""Model training module."""

import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from typing import Optional, Dict, Any
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import time

from ..data.utils import normalize_team_name

logger = logging.getLogger(__name__)


def load_and_prepare_data(
    data_file: Path,
    balance_classes: bool = True,
    test_size: float = 0.2,
    random_seed: int = 49,
    verbose: bool = True
) -> tuple:
    """Load and prepare data for training.
    
    Args:
        data_file: Path to final_features.csv.
        balance_classes: Whether to balance classes by undersampling.
        test_size: Fraction of data to use for test set (default: 0.2 for 80/20 split).
        random_seed: Random seed for reproducibility.
        verbose: Whether to print progress messages.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    np.random.seed(random_seed)
    
    if verbose:
        logger.info("Loading data...")
    
    df = pd.read_csv(data_file)
    
    if verbose:
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
    
    target_col = 'team_a_won'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if verbose:
        logger.info(f"\nFeatures: {X.shape[1]} columns")
        logger.info(f"Original target distribution:\n{y.value_counts()}")
    
    # Balance classes if requested
    if balance_classes:
        if verbose:
            logger.info("\nBALANCING CLASSES")
        
        df_combined = pd.concat([X, y], axis=1)
        class_counts = y.value_counts()
        minority_count = class_counts.min()
        
        df_minority = df_combined[df_combined[target_col] == class_counts.idxmin()]
        df_majority = df_combined[df_combined[target_col] == class_counts.idxmax()]
        df_majority_balanced = df_majority.sample(n=minority_count, random_state=random_seed)
        
        df_balanced = pd.concat([df_minority, df_majority_balanced], ignore_index=True)
        df_balanced = df_balanced.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        X = df_balanced.drop(columns=[target_col])
        y = df_balanced[target_col]
        
        if verbose:
            logger.info(f"After balancing: {len(df_balanced)} samples")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    
    if verbose:
        logger.info(f"\nTrain set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    hyperparameter_tuning: bool = True,
    n_iter: int = 50,
    random_seed: int = 49,
    verbose: bool = True
) -> tuple:
    """Train XGBoost model with optional hyperparameter tuning.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Optional test features for early stopping.
        y_test: Optional test target for early stopping.
        hyperparameter_tuning: Whether to perform hyperparameter tuning.
        n_iter: Number of random hyperparameter combinations to try.
        random_seed: Random seed for reproducibility.
        verbose: Whether to print progress messages.
        
    Returns:
        Tuple of (trained_model, best_params_dict).
    """
    np.random.seed(random_seed)
    
    # Base parameters
    base_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': random_seed,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    
    if verbose:
        logger.info("XGBOOST MODEL TRAINING")
    
    if hyperparameter_tuning:
        if verbose:
            logger.info("\nPerforming hyperparameter tuning...")
        
        param_grid = {
            'max_depth': [4, 5, 6, 7, 8, 10, 20],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'n_estimators': [100, 200, 300, 500],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.5, 1.0, 1.5, 2.0],
            'scale_pos_weight': [1]
        }
        
        base_model = xgb.XGBClassifier(**base_params)
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='f1',
            n_jobs=-1,
            cv=3,
            verbose=2 if verbose else 0,
            random_state=random_seed,
            return_train_score=True
        )
        
        start_time = time.time()
        random_search.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        
        if verbose:
            logger.info(f"\nHyperparameter search completed in {elapsed_time:.2f} seconds")
            logger.info(f"Best parameters: {random_search.best_params_}")
            logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        
        best_params = random_search.best_params_.copy()
        best_params.update(base_params)  # Ensure base params are included
        best_params['random_state'] = random_seed  # Explicitly set random seed
        # Create a new model with best parameters (don't reuse best_estimator_)
        model = xgb.XGBClassifier(**best_params)
    else:
        best_params = base_params
        model = xgb.XGBClassifier(**best_params)
    
    # Final training with early stopping if test set provided
    if X_test is not None and y_test is not None:
        if verbose:
            logger.info("\nFinal training with early stopping...")
        
        model.set_params(early_stopping_rounds=50)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=50 if verbose else False
        )
    else:
        model.fit(X_train, y_train)
    
    if verbose:
        logger.info("\nModel training completed!")
    
    return model, best_params
