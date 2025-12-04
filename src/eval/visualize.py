"""Visualization utilities for model evaluation."""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    roc_auc_score
)

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """Plot confusion matrix as a heatmap.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Team A Lost', 'Team A Won'],
        yticklabels=['Team A Lost', 'Team A Won'],
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """Plot ROC curve.
    
    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curve to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """Plot Precision-Recall curve.
    
    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Precision-Recall curve to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 20,
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """Plot feature importance from XGBoost model.
    
    Args:
        model: Trained XGBoost model.
        feature_names: List of feature names.
        top_n: Number of top features to display.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.
    """
    import xgboost as xgb
    
    if not isinstance(model, xgb.XGBClassifier):
        logger.warning("Feature importance only available for XGBoost models")
        return
    
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importance[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_history(
    eval_results: dict,
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """Plot training history from XGBoost eval_set results.
    
    Args:
        eval_results: Dictionary with 'train' and 'test' keys containing lists of scores.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.
    """
    if 'train' not in eval_results or 'test' not in eval_results:
        logger.warning("Training history requires 'train' and 'test' keys in eval_results")
        return
    
    train_scores = eval_results['train']
    test_scores = eval_results['test']
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_scores, label='Train', marker='o')
    plt.plot(test_scores, label='Test', marker='s')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Log Loss', fontsize=12)
    plt.title('Training History (Log Loss)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_all_visualizations(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
    show: bool = False
) -> None:
    """Create all visualizations for model evaluation.
    
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: True test labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (optional).
        output_dir: Directory to save plots.
        show: Whether to display plots.
    """
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating visualizations...")
    
    # Confusion Matrix
    cm_path = output_dir / "confusion_matrix.png" if output_dir else None
    plot_confusion_matrix(y_test, y_pred, save_path=cm_path, show=show)
    
    # ROC Curve (if probabilities available)
    if y_proba is not None:
        roc_path = output_dir / "roc_curve.png" if output_dir else None
        plot_roc_curve(y_test, y_proba, save_path=roc_path, show=show)
        
        pr_path = output_dir / "precision_recall_curve.png" if output_dir else None
        plot_precision_recall_curve(y_test, y_proba, save_path=pr_path, show=show)
    
    # Feature Importance (if XGBoost model)
    try:
        import xgboost as xgb
        if isinstance(model, xgb.XGBClassifier):
            fi_path = output_dir / "feature_importance.png" if output_dir else None
            plot_feature_importance(
                model,
                feature_names=list(X_test.columns),
                save_path=fi_path,
                show=show
            )
    except ImportError:
        pass
    
    logger.info("Visualizations created successfully")

