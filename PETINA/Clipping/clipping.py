# Clipping Module for PETINA
import numpy as np
from PETINA.Data_Conversion_Helper import type_checking_and_return_lists, type_checking_return_actual_dtype

# -------------------------------
# Simple Clipping
# -------------------------------
def applyClipping(values, clipping_threshold):
    """
    Clips values at the given clipping threshold.

    Parameters:
        values (list or np.array): List or array of numerical values.
        clipping_threshold (float): The max threshold for clipping.

    Returns:
        List of clipped values.
    """
    values = np.array(values)
    clipped = np.minimum(values, clipping_threshold)
    return clipped.tolist()


# -------------------------------
# Adaptive Clipping based on quantile
# -------------------------------
def applyClippingAdaptive(domain):
    """
    Applies adaptive clipping using the 5th percentile as lower bound
    and the max value as upper bound.

    Parameters:
        domain: Input data (list, numpy array, or tensor).

    Returns:
        Data with adaptive clipping applied in original format.
    """
    values, shape = type_checking_and_return_lists(domain)
    lower_quantile = 0.05
    lower_bound = np.quantile(values, lower_quantile)
    upper_bound = np.max(values)

    clipped = np.clip(values, lower_bound, upper_bound)
    return type_checking_return_actual_dtype(domain, clipped.tolist(), shape)


# -------------------------------
# Clipping with Differential Privacy (Laplace noise)
# -------------------------------
def applyClippingDP(domain, clipping_threshold, sensitivity, epsilon):
    """
    Clips values then adds Laplace noise for differential privacy.

    Parameters:
        domain: Input data (list, numpy array, or tensor).
        clipping_threshold (float): Clipping threshold.
        sensitivity (float): Sensitivity of the data.
        epsilon (float): Privacy budget.

    Returns:
        Differentially private clipped data in original format.
    """
    values, shape = type_checking_and_return_lists(domain)
    clipped_values = applyClipping(values, clipping_threshold)
    noise_scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=noise_scale, size=len(clipped_values))
    privatized = np.array(clipped_values) + noise
    return type_checking_return_actual_dtype(domain, privatized.tolist(), shape)
