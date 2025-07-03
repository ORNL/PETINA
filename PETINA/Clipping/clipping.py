import math
import random
import torch
import numpy as np
from scipy import stats as st

from PETINA.Data_Conversion_Helper import type_checking_and_return_lists, type_checking_return_actual_dtype

# -------------------------------
# Clipping Functions
# Source: https://arxiv.org/pdf/2311.06839.pdf
# Implementation: https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
# -------------------------------
def applyClipping(value, clipping):
    """
    Applies simple clipping to each element in the list.
    If a value is above the clipping threshold, it is set to the threshold.

    Parameters:
        value (list): A list of numerical values.
        clipping (float): The clipping threshold.

    Returns:
        A list of clipped values.
    """
    clipped = []
    for i in range(len(value)):
        if value[i] >= clipping:
            clipped.append(clipping)
        else:
            clipped.append(value[i])
    return clipped


# -------------------------------
# Source: https://arxiv.org/pdf/2311.06839.pdf
# -------------------------------
def applyClippingAdaptive(domain):
    """
    Applies adaptive clipping based on the lower 5th percentile of the data.
    This ensures that the lower tail of the distribution is used as a clipping threshold.

    Parameters:
        domain: Input data (list, array-like, or tensor).

    Returns:
        Data with adaptive clipping applied, in the same format as the input.
    """
    value, shape = type_checking_and_return_lists(domain)

    lower_quantile = 0.05
    lower = np.quantile(value, lower_quantile)

    # Clip values between the lower bound and the maximum value.
    clipped_data = np.clip(value, lower, np.max(value))
    clipped_data = clipped_data.tolist()
    return type_checking_return_actual_dtype(domain, clipped_data, shape)


# -------------------------------
# Clipping Functions
# Source: https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
# Implementation: https://medium.com/pytorch/differential-privacy-series-part-1-dp-sgd-algorithm-explained-12512c3959a3
# -------------------------------
def applyClippingDP(domain, clipping, sensitivity, epsilon):
    """
    Applies clipping with differential privacy.
    First, values are clipped; then Laplace noise is added.

    Parameters:
        domain: Input data (list, numpy array, or tensor).
        clipping (float): Clipping threshold.
        sensitivity (float): Sensitivity of the data.
        epsilon (float): Privacy parameter.

    Returns:
        Data with differentially private clipping applied.
    """
    value, shape = type_checking_and_return_lists(domain)
    tmpValue = applyClipping(value, clipping)
    privatized = []
    for i in range(len(tmpValue)):
        privatized.append(tmpValue[i] + np.random.laplace(loc=0, scale=sensitivity / epsilon))
    return type_checking_return_actual_dtype(domain, privatized, shape)