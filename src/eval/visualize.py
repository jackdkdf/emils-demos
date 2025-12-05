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


def plot_class_distribution(
    original_distribution: dict,
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """Plot class distribution before and after balancing.
    
    Args:
        original_distribution: Dictionary with 'before' and 'after' keys containing class counts.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.
    """
    if 'before' not in original_distribution or 'after' not in original_distribution:
        logger.warning("Class distribution requires 'before' and 'after' keys")
        return
    
    before_counts = original_distribution['before']
    after_counts = original_distribution['after']
    
    # Prepare data for plotting
    classes = sorted(set(list(before_counts.keys()) + list(after_counts.keys())))
    class_labels = ['Team A Lost', 'Team A Won']  # Assuming binary classification
    
    before_values = [before_counts.get(cls, 0) for cls in classes]
    after_values = [after_counts.get(cls, 0) for cls in classes]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Before balancing
    bars1 = ax1.bar(class_labels, before_values, color=['#3498db', '#e74c3c'], alpha=0.8)
    ax1.set_title('Class Distribution Before Balancing', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_xlabel('Class', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/sum(before_values)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    # After balancing
    bars2 = ax2.bar(class_labels, after_values, color=['#3498db', '#e74c3c'], alpha=0.8)
    ax2.set_title('Class Distribution After Balancing', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_xlabel('Class', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/sum(after_values)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved class distribution plot to: {save_path}")
    
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


def plot_loss_curves(
    eval_results: dict,
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """Plot loss curves from XGBoost eval_set results.
    
    Args:
        eval_results: Dictionary from XGBoost evals_result() method.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.
    """
    if not eval_results:
        logger.warning("No evaluation results available for loss curves")
        return
    
    # XGBoost stores results as {'validation_0': {'logloss': [...]}, 'validation_1': {'logloss': [...]}}
    # validation_0 is typically train, validation_1 is test
    train_key = 'validation_0'
    test_key = 'validation_1'
    
    if train_key not in eval_results or test_key not in eval_results:
        logger.warning("Training history requires validation_0 and validation_1 keys")
        return
    
    train_loss = eval_results[train_key].get('logloss', [])
    test_loss = eval_results[test_key].get('logloss', [])
    
    if not train_loss or not test_loss:
        logger.warning("No logloss data found in evaluation results")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss', marker='o', markersize=4)
    plt.plot(test_loss, label='Validation Loss', marker='s', markersize=4)
    plt.xlabel('Boosting Round', fontsize=12)
    plt.ylabel('Log Loss', fontsize=12)
    plt.title('Training Loss Curves', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved loss curves to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_accuracy_curves(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """Plot accuracy curves by evaluating model at different boosting rounds.
    
    Args:
        model: Trained XGBoost model.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.
    """
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    
    if not isinstance(model, xgb.XGBClassifier):
        logger.warning("Accuracy curves only available for XGBoost models")
        return
    
    # Get the number of boosting rounds from the model
    # XGBoost models store the actual number of trees used
    try:
        n_estimators = model.get_booster().num_boosted_rounds()
    except:
        n_estimators = model.get_params().get('n_estimators', 100)
    
    # Limit to reasonable number of evaluation points
    max_points = 50
    step = max(1, n_estimators // max_points)
    eval_rounds = list(range(step, n_estimators + 1, step))
    if eval_rounds[-1] != n_estimators:
        eval_rounds.append(n_estimators)
    
    train_accuracies = []
    test_accuracies = []
    
    # Evaluate model at different stages using iteration_range
    for i in eval_rounds:
        # Get predictions up to iteration i
        train_pred_proba = model.predict_proba(X_train, iteration_range=(0, i))[:, 1]
        test_pred_proba = model.predict_proba(X_test, iteration_range=(0, i))[:, 1]
        
        # Convert probabilities to binary predictions
        train_pred = (train_pred_proba >= 0.5).astype(int)
        test_pred = (test_pred_proba >= 0.5).astype(int)
        
        # Calculate accuracies
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    
    plt.figure(figsize=(10, 6))
    plt.plot(eval_rounds, train_accuracies, label='Train Accuracy', marker='o', markersize=4)
    plt.plot(eval_rounds, test_accuracies, label='Validation Accuracy', marker='s', markersize=4)
    plt.xlabel('Boosting Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training Accuracy Curves', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved accuracy curves to: {save_path}")
    
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
    show: bool = False,
    eval_results: Optional[dict] = None,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    original_distribution: Optional[dict] = None
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
        eval_results: Optional evaluation results from XGBoost for loss curves.
        X_train: Optional training features for accuracy curves.
        y_train: Optional training labels for accuracy curves.
        original_distribution: Optional dictionary with 'before' and 'after' class distributions.
    """
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating visualizations...")
    
    # Class Distribution (if available)
    if original_distribution is not None:
        dist_path = output_dir / "class_distribution.png" if output_dir else None
        plot_class_distribution(original_distribution, save_path=dist_path, show=show)
    
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
            
            # Loss curves (if eval_results available)
            if eval_results is not None:
                loss_path = output_dir / "loss_curves.png" if output_dir else None
                plot_loss_curves(eval_results, save_path=loss_path, show=show)
            
            # Accuracy curves (if training data available)
            if X_train is not None and y_train is not None:
                acc_path = output_dir / "accuracy_curves.png" if output_dir else None
                plot_accuracy_curves(
                    model, X_train, y_train, X_test, y_test,
                    save_path=acc_path, show=show
                )
    except ImportError:
        pass
    
    logger.info("Visualizations created successfully")

