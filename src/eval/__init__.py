"""Evaluation modules."""

from .metrics import evaluate_model
from .visualize import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance,
    plot_loss_curves,
    plot_accuracy_curves,
    plot_class_distribution,
    create_all_visualizations
)

__all__ = [
    'evaluate_model',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_feature_importance',
    'plot_loss_curves',
    'plot_accuracy_curves',
    'plot_class_distribution',
    'analyze_prediction_errors',
    'create_all_visualizations',
]
