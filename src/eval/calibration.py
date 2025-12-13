"""Calibration metrics for model predictions."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


def calculate_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 50
) -> Dict:
    """Calculate calibration metrics by binning predictions.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities for positive class
        n_bins: Number of bins to use for calibration
        
    Returns:
        Dictionary with:
        - bins: List of bin edges
        - bin_centers: Center of each bin
        - bin_counts: Number of samples in each bin
        - bin_accuracies: Accuracy for each bin
        - bin_mean_proba: Mean predicted probability in each bin
    """
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bins) - 1
    # Handle edge case where probability is exactly 1.0
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Initialize arrays for ALL bins (even empty ones) to maintain index alignment
    bin_centers = []
    bin_counts = []
    bin_accuracies = []
    bin_mean_proba = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        bin_count = np.sum(mask)
        
        # Always calculate bin center (even for empty bins) to maintain index alignment
        bin_center = (bins[i] + bins[i+1]) / 2
        
        if bin_count == 0:
            # Empty bin: use overall accuracy as placeholder (will be smoothed later)
            bin_centers.append(bin_center)
            bin_counts.append(0)
            bin_accuracies.append(None)  # Will be filled with overall accuracy during smoothing
            bin_mean_proba.append(bin_center)  # Use bin center as placeholder
            continue
        
        bin_true = y_true[mask]
        bin_proba = y_proba[mask]
        
        # Calculate accuracy for this bin
        # Accuracy = fraction of predictions where predicted class matches true class
        # For each sample in this bin, check if the prediction (based on prob > 0.5) matches the true label
        bin_pred = (bin_proba > 0.5).astype(int)
        bin_accuracy = np.mean(bin_pred == bin_true)
        
        # Also calculate the fraction of positive class in this bin (for calibration check)
        # This should be close to the mean probability if well-calibrated
        fraction_positive = np.mean(bin_true)
        mean_proba = np.mean(bin_proba)
        
        # The calibration accuracy should reflect: "When model predicts prob in this range,
        # how often is the predicted class correct?"
        # For a well-calibrated model:
        # - If mean_proba > 0.5: we predict class 1, accuracy ≈ fraction_positive
        # - If mean_proba < 0.5: we predict class 0, accuracy ≈ 1 - fraction_positive
        # But we use actual bin_accuracy which accounts for individual predictions
        
        # Store fraction positive for debugging/validation
        if 'fraction_positive_list' not in locals():
            fraction_positive_list = []
        fraction_positive_list.append(fraction_positive)
        
        bin_centers.append(bin_center)
        bin_counts.append(bin_count)
        bin_accuracies.append(bin_accuracy)
        bin_mean_proba.append(mean_proba)
    
    # Calculate overall accuracy for reference
    overall_accuracy = np.mean((y_proba > 0.5).astype(int) == y_true)
    
    # Smooth bin accuracies towards overall accuracy for bins with few samples
    # This prevents low accuracy estimates for bins with insufficient data
    # Use a more conservative threshold: need at least 20 samples or 1/10th of average per bin
    min_samples_for_reliable = max(20, len(y_true) // (n_bins * 10))
    smoothed_accuracies = []
    
    for i, (acc, count, center) in enumerate(zip(bin_accuracies, bin_counts, bin_centers)):
        # Handle empty bins (acc is None)
        if acc is None or count == 0:
            smoothed_accuracies.append(overall_accuracy)
            continue
        
        if count < min_samples_for_reliable:
            # For bins with few samples, blend with overall accuracy
            # Weight: more samples = more trust in bin accuracy, fewer samples = more trust in overall
            # Use a more conservative smoothing: only smooth if we have very few samples
            if count < 5:
                # Very few samples: use overall accuracy
                smoothed_acc = overall_accuracy
            elif count < 10:
                # Very few samples: heavily weight towards overall accuracy
                sample_weight = count / 10.0  # 0.0 to 1.0
                smoothed_acc = acc * sample_weight + overall_accuracy * (1 - sample_weight)
            else:
                # Some samples: blend between bin accuracy and overall accuracy
                sample_weight = min(1.0, count / min_samples_for_reliable)
                smoothed_acc = acc * sample_weight + overall_accuracy * (1 - sample_weight)
            smoothed_accuracies.append(smoothed_acc)
        else:
            # For bins with enough samples, use the bin accuracy directly
            # This is the most accurate representation of model performance for this probability range
            # Don't smooth - use actual measured accuracy
            smoothed_accuracies.append(acc)
    
    # Validate calibration: weighted average of bin accuracies should be close to overall accuracy
    total_samples = sum(bin_counts)
    if total_samples > 0:
        weighted_avg_accuracy = sum(acc * count for acc, count in zip(smoothed_accuracies, bin_counts)) / total_samples
        calibration_error = abs(weighted_avg_accuracy - overall_accuracy)
        
        if calibration_error > 0.05:  # More than 5% difference
            logger.warning(
                f"Calibration validation: Weighted average bin accuracy ({weighted_avg_accuracy:.2%}) "
                f"differs from overall accuracy ({overall_accuracy:.2%}) by {calibration_error:.2%}. "
                f"This may indicate smoothing is too aggressive."
            )
    
    return {
        'bins': bins.tolist(),
        'bin_centers': bin_centers,
        'bin_counts': bin_counts,
        'bin_accuracies': smoothed_accuracies,  # Use smoothed accuracies
        'bin_mean_proba': bin_mean_proba,
        'overall_accuracy': float(overall_accuracy),
        'weighted_avg_accuracy': float(weighted_avg_accuracy) if total_samples > 0 else None,
    }


def get_calibration_accuracy(
    calibration_data: Dict,
    predicted_probability: float,
    use_interpolation: bool = True
) -> Optional[float]:
    """Get the calibration accuracy for a given predicted probability.
    
    Args:
        calibration_data: Calibration metrics dictionary from calculate_calibration_metrics
        predicted_probability: The predicted probability to look up
        use_interpolation: If True, interpolate between bins for smoother results
        
    Returns:
        Calibration accuracy for the probability bin (or interpolated), or None if not found
    """
    if not calibration_data or 'bins' not in calibration_data:
        return None
    
    bins = np.array(calibration_data['bins'])
    bin_accuracies = calibration_data['bin_accuracies']
    bin_centers = np.array(calibration_data['bin_centers'])
    bin_counts = calibration_data['bin_counts']
    
    if len(bin_accuracies) == 0:
        return None
    
    # Find which bin this probability falls into
    bin_idx = np.digitize([predicted_probability], bins)[0] - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_accuracies) - 1)
    
    if use_interpolation and len(bin_accuracies) > 1:
        # Use linear interpolation between bins for smoother results
        # But be conservative - only interpolate if bins have enough samples
        min_samples_for_interpolation = 10
        
        if bin_idx == 0:
            # At the start, use the first bin
            return bin_accuracies[0]
        elif bin_idx >= len(bin_accuracies) - 1:
            # At the end, use the last bin
            return bin_accuracies[-1]
        else:
            # Check if both bins have enough samples for reliable interpolation
            count_prev = bin_counts[bin_idx - 1]
            count_next = bin_counts[bin_idx]
            
            # If either bin has very few samples, don't interpolate - use the bin with more samples
            if count_prev < min_samples_for_interpolation or count_next < min_samples_for_interpolation:
                if count_prev >= count_next:
                    return bin_accuracies[bin_idx - 1]
                else:
                    return bin_accuracies[bin_idx]
            
            # Both bins have enough samples - safe to interpolate
            center_prev = bin_centers[bin_idx - 1]
            center_next = bin_centers[bin_idx]
            
            if center_next == center_prev:
                return bin_accuracies[bin_idx]
            
            # Linear interpolation weight
            weight = (predicted_probability - center_prev) / (center_next - center_prev)
            weight = np.clip(weight, 0, 1)
            
            # Weighted average of accuracies, also consider bin counts for reliability
            total_count = count_prev + count_next
            
            if total_count > 0:
                # Weight by both distance and sample count
                weight_prev = (1 - weight) * (count_prev / total_count)
                weight_next = weight * (count_next / total_count)
                total_weight = weight_prev + weight_next
                
                if total_weight > 0:
                    interpolated = (bin_accuracies[bin_idx - 1] * weight_prev + 
                                   bin_accuracies[bin_idx] * weight_next) / total_weight
                    # Cap interpolation to not exceed reasonable bounds (don't go too far from actual bin values)
                    # This prevents interpolation from creating unrealistic accuracy values
                    min_acc = min(bin_accuracies[bin_idx - 1], bin_accuracies[bin_idx])
                    max_acc = max(bin_accuracies[bin_idx - 1], bin_accuracies[bin_idx])
                    # Only allow interpolation within the range of the two bins (no extrapolation)
                    interpolated = np.clip(interpolated, min_acc, max_acc)
                    return interpolated
            
            # Fallback to simple interpolation
            interpolated = bin_accuracies[bin_idx - 1] * (1 - weight) + bin_accuracies[bin_idx] * weight
            return interpolated
    
    # Simple bin lookup
    if bin_idx < len(bin_accuracies):
        return bin_accuracies[bin_idx]
    
    return None

