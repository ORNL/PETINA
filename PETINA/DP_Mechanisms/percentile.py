import torch
import numpy as np
from scipy import stats as st

from PETINA.Data_Conversion_Helper import type_checking_and_return_lists,type_checking_return_actual_dtype

# -------------------------------
# Source: Smith, A. (2011, June). Privacy-preserving statistical estimation with optimal convergence rates.
# In Proceedings of the forty-third annual ACM symposium on Theory of computing (pp. 813-822).
# -------------------------------
def percentilePrivacy(domain, percentile):
    """
    Applies percentile privacy by setting values below a specified percentile to zero.

    Parameters:
        domain: Input data (list, numpy array, or tensor).
        percentile (float): Lower percentile threshold (0-100).

    Returns:
        Data with values below the percentile set to zero, in the same format as the input.
    """
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100.")

    data, shape = type_checking_and_return_lists(domain)
    data = np.array(data)

    # Determine the lower bound using the percentile.
    lower_bound = np.percentile(data, percentile)

    # Replace values below the lower bound with zero.
    data = np.where((data >= lower_bound), data, 0)
    data = data.tolist()
    return type_checking_return_actual_dtype(domain, data, shape)