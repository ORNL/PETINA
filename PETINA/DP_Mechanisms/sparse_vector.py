import math
import random
import torch
import numpy as np
from scipy import stats as st

from PETINA.Data_Conversion_Helper import type_checking_and_return_lists

# -------------------------------
# Sparse Vector Technique (SVT)
# Source: Cynthia Dwork, Aaron Roth, and others. The algorithmic foundations of differential privacy. Foundations and Trends® in Theoretical Computer Science, 9(3–4):211–407, 2014.
# -------------------------------
def above_threshold_SVT(val, domain, T, epsilon):
    """
    Implements the Sparse Vector Technique (SVT) for differential privacy.
    Returns the actual value if the noisy value exceeds a threshold; otherwise,
    returns a random value from the domain.

    Parameters:
        val (float): The value to check against the threshold.
        domain: Input data (list, numpy array, or tensor) used as a fallback.
        T (float): The threshold value.
        epsilon (float): Privacy parameter.

    Returns:
        The original value if the condition is met; otherwise, a random value from domain.
    """
    possible_val_list, shape = type_checking_and_return_lists(domain)
    T_hat = T + np.random.laplace(loc=0, scale=2 / epsilon)  # Noisy threshold

    nu_i = np.random.laplace(loc=0, scale=4 / epsilon)  # Noise added to the value
    if val + nu_i >= T_hat:
        return val
    # Fallback: return a random value if the threshold condition is not met.
    return random.choice(possible_val_list)
