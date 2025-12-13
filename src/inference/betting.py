"""Betting calculations for match predictions."""

from typing import Optional, Tuple, Dict


def american_to_decimal(american_odds: str) -> Optional[float]:
    """Convert American odds format to decimal odds.
    
    Args:
        american_odds: American odds as string (e.g., "+100", "-120", "+150")
        
    Returns:
        Decimal odds (e.g., 2.0, 1.83, 2.5) or None if invalid
    """
    try:
        # Remove any whitespace and + sign
        odds_str = american_odds.strip().replace('+', '')
        
        # Parse as integer
        odds_int = int(odds_str)
        
        if odds_int > 0:
            # Positive odds: decimal = (american / 100) + 1
            return (odds_int / 100.0) + 1.0
        else:
            # Negative odds: decimal = (100 / abs(american)) + 1
            return (100.0 / abs(odds_int)) + 1.0
    except (ValueError, ZeroDivisionError):
        return None


def calculate_expected_value(
    model_probability: float,
    decimal_odds: float,
    calibration_accuracy: Optional[float] = None
) -> Tuple[float, float]:
    """Calculate expected value and profit for a bet.
    
    Args:
        model_probability: Model's predicted probability (0-1)
        decimal_odds: Decimal betting odds (e.g., 2.0 for even money)
        calibration_accuracy: Model's calibration accuracy for this probability range (optional, for info only)
        
    Returns:
        Tuple of (expected_value, expected_profit_per_unit_bet)
        Expected value: positive means profitable bet, negative means unprofitable
        Expected profit: average profit per unit bet
        
    Note:
        Calibration accuracy is NOT used as a probability replacement. It's only provided
        for informational purposes. Expected value is calculated using the model's predicted
        probability directly.
    """
    # Use model probability directly for EV calculation
    # Calibration accuracy is informational only (shows model reliability)
    # Expected value = (probability * (odds - 1)) - (1 - probability) * 1
    # Simplified: probability * odds - 1
    expected_value = (model_probability * decimal_odds) - 1.0
    
    # Expected profit per unit bet (same as expected value)
    expected_profit = expected_value
    
    return expected_value, expected_profit


def calculate_betting_recommendation(
    team_a_prob: float,
    team_b_prob: float,
    team_a_odds: float,
    team_b_odds: float,
    team_a_calib_acc: Optional[float] = None,
    team_b_calib_acc: Optional[float] = None
) -> Dict:
    """Calculate betting recommendations based on model predictions and odds.
    
    Args:
        team_a_prob: Model probability for team A winning
        team_b_prob: Model probability for team B winning
        team_a_odds: Decimal betting odds for team A
        team_b_odds: Decimal betting odds for team B
        team_a_calib_acc: Calibration accuracy for team A probability
        team_b_calib_acc: Calibration accuracy for team B probability
        
    Returns:
        Dictionary with betting recommendations:
        {
            'team_a': {
                'ev': expected_value,
                'expected_profit': expected_profit_per_unit,
                'odds': decimal_odds,
                'recommendation': 'bet' | 'avoid'
            },
            'team_b': { ... },
            'best_bet': 'team_a' | 'team_b' | None
        }
    """
    recommendations = {}
    
    # Calculate for team A
    ev_a, profit_a = calculate_expected_value(
        team_a_prob, team_a_odds, team_a_calib_acc
    )
    recommendations['team_a'] = {
        'ev': ev_a,
        'expected_profit': profit_a,
        'odds': team_a_odds,
        'model_probability': team_a_prob,
        'calibration_accuracy': team_a_calib_acc,
        'recommendation': 'bet' if ev_a > 0 else 'avoid'
    }
    
    # Calculate for team B
    ev_b, profit_b = calculate_expected_value(
        team_b_prob, team_b_odds, team_b_calib_acc
    )
    recommendations['team_b'] = {
        'ev': ev_b,
        'expected_profit': profit_b,
        'odds': team_b_odds,
        'model_probability': team_b_prob,
        'calibration_accuracy': team_b_calib_acc,
        'recommendation': 'bet' if ev_b > 0 else 'avoid'
    }
    
    # Determine best bet
    if ev_a > 0 and ev_b > 0:
        recommendations['best_bet'] = 'team_a' if ev_a > ev_b else 'team_b'
    elif ev_a > 0:
        recommendations['best_bet'] = 'team_a'
    elif ev_b > 0:
        recommendations['best_bet'] = 'team_b'
    else:
        recommendations['best_bet'] = None
    
    return recommendations

