# Clipping Module for PETINA
import numpy as np
# from PETINA.Data_Conversion_Helper import type_checking_and_return_lists, type_checking_return_actual_dtype
from PETINA.Data_Conversion_Helper import TypeConverter
# -------------------------------
# Simple Clipping
# -------------------------------
def applyClipping(values, lower_bound, upper_bound):
    """
    Clips values using fixed lower and upper bounds.

    Parameters:
        values (list or np.array): List or array of numerical values.
        lower_bound (float): Minimum value to clip to.
        upper_bound (float): Maximum value to clip to.

    Returns:
        List of values clipped to the specified range.
    """
    # Convert input to NumPy array for vectorized operations
    values = np.array(values)

    # Clip values between the specified bounds
    clipped = np.clip(values, lower_bound, upper_bound)

    # Convert back to list and return
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
        Data with adaptive clipping applied, in the same format as the input.
    """
    # Flatten and track type
    converter = TypeConverter(domain)
    values, _ = converter.get()

    # Compute clipping bounds
    lower_quantile = 0.05
    lower_bound = np.quantile(values, lower_quantile)
    upper_bound = np.max(values)

    # Apply clipping
    clipped = np.clip(values, lower_bound, upper_bound)

    # Restore original type/shape
    return converter.restore(clipped.tolist())

# -------------------------------
# Clipping with Differential Privacy (Laplace noise)
# -------------------------------
def applyClippingDP(domain, sensitivity, epsilon, lower_quantile=0.05, upper_quantile=0.95):
    """
    Clips values between dynamically chosen bounds (percentiles), then adds Laplace noise for DP.

    Parameters:
        domain: Input data (list, numpy array, or tensor).
        sensitivity (float): Sensitivity of the data.
        epsilon (float): Privacy budget.
        lower_quantile (float): Lower quantile for clipping bound (default 5th percentile).
        upper_quantile (float): Upper quantile for clipping bound (default 95th percentile).

    Returns:
        Differentially private clipped data in the original format.
    """
    # Flatten and track type
    converter = TypeConverter(domain)
    values, _ = converter.get()

    # Compute clipping bounds dynamically
    lower_bound = np.quantile(values, lower_quantile)
    upper_bound = np.quantile(values, upper_quantile)

    # Apply clipping with dynamic bounds
    clipped_values = applyClipping(values, lower_bound, upper_bound)

    # Add Laplace noise
    noise_scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=noise_scale, size=len(clipped_values))
    privatized = np.array(clipped_values) + noise

    # Restore to original input type
    return converter.restore(privatized.tolist())