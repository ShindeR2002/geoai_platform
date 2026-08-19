import numpy as np
import scipy.stats as stats
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

def run_wilcoxon_signed_rank_test(scores_a: list, scores_b: list) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test to determine if there is a statistically
    significant difference between two paired metric distributions across datasets/runs.
    Returns:
        statistic (float): Wilcoxon test statistic.
        p_value (float): The two-sided p-value.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"Sample size mismatch for Wilcoxon test: {len(scores_a)} vs {len(scores_b)}")
        
    diff = np.array(scores_a) - np.array(scores_b)
    # Wilcoxon requires non-zero differences
    if np.all(diff == 0):
        return 0.0, 1.0 # Identical distributions
        
    try:
        statistic, p_value = stats.wilcoxon(scores_a, scores_b)
        return float(statistic), float(p_value)
    except Exception as e:
        logger.warning("Wilcoxon signed-rank test failed: %s. Returning defaults.", e)
        return 0.0, 1.0

def run_mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Tuple[float, float]:
    """
    Perform McNemar's test on pixel prediction contingency table:
    
                       Model B Correct | Model B Incorrect
    Model A Correct           a        |        b
    Model A Incorrect         c        |        d
    
    Test statistic: chi2 = (|b - c| - 1)**2 / (b + c) with Edwards continuity correction.
    If b + c < 25, exact binomial test is used.
    """
    # 1. Compute cell counts
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    
    # b: A correct, B incorrect
    b = int(np.sum(correct_a & ~correct_b))
    # c: A incorrect, B correct
    c = int(np.sum(~correct_a & correct_b))
    
    if (b + c) == 0:
        return 0.0, 1.0 # Identical performance distributions
        
    if (b + c) < 25:
        # Use exact binomial test
        # Null hypothesis: p = 0.5. Successes = min(b, c), Trials = b + c
        p_value = 2.0 * stats.binom.cdf(min(b, c), b + c, 0.5)
        p_value = min(1.0, p_value)
        return 0.0, float(p_value)
    else:
        # Chi-squared approximation with continuity correction
        chi2 = (abs(b - c) - 1.0)**2 / (b + c)
        p_value = stats.chi2.sf(chi2, df=1)
        return float(chi2), float(p_value)
