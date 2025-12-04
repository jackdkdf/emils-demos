"""Evaluation metrics and utilities."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, adjusted_rand_score
)

logger = logging.getLogger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    verbose: bool = True
) -> Dict[str, float]:
    """Evaluate model predictions.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        verbose: Whether to print metrics.
        
    Returns:
        Dictionary of metric names and values.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    ari = adjusted_rand_score(y_true.flatten(), y_pred.flatten())
    
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'adjusted_rand_index': ari,
        'confusion_matrix': cm
    }
    
    if verbose:
        logger.info("MODEL EVALUATION")
        logger.info(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        logger.info(f"Precision: {prec:.4f}")
        logger.info(f"Recall:    {rec:.4f}")
        logger.info(f"F1-Score:  {f1:.4f}")
        logger.info(f"ARI:       {ari:.4f}")
        logger.info("")
        logger.info("Confusion Matrix:")
        logger.info(f"                Predicted")
        logger.info(f"                0      1")
        logger.info(f"Actual  0    {cm[0,0]:5d}  {cm[0,1]:5d}")
        logger.info(f"        1    {cm[1,0]:5d}  {cm[1,1]:5d}")
        logger.info("")
        logger.info(f"True Negatives:  {cm[0,0]}")
        logger.info(f"False Positives: {cm[0,1]}")
        logger.info(f"False Negatives: {cm[1,0]}")
        logger.info(f"True Positives:  {cm[1,1]}")
    
    return metrics
