import math
import random
import torch
import numpy as np
from scipy import stats as st

from PETINA.Data_Conversion_Helper import type_checking_and_return_lists, type_checking_return_actual_dtype
# -------------------------------
# Pruning Functions
# Source: https://arxiv.org/pdf/2311.06839.pdf
# Implementation: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700324.pdf
# -------------------------------
def applyPruning(domain, prune_ratio):
    """
    Applies pruning to reduce the magnitude of values.
    Values with an absolute value below the prune_ratio may be set to 0 or pruned to the prune_ratio.

    Parameters:
        domain: Input data (list, numpy array, or tensor).
        prune_ratio (float): Threshold below which values are pruned.

    Returns:
        Pruned data in the same format as the input.
    """
    value, shape = type_checking_and_return_lists(domain)
    pruned = []
    for i in range(len(value)):
        if abs(value[i]) < prune_ratio:
            rnd_tmp = random.random()
            if abs(value[i]) > rnd_tmp * prune_ratio:
                # Set to prune_ratio preserving the sign.
                if value[i] > 0:
                    pruned.append(prune_ratio)
                else:
                    pruned.append(-prune_ratio)
            else:
                pruned.append(0)
    print("domain", domain)  # Jackie comment: This is for debugging purposes, can be removed later
    print("pruned", pruned)  # Jackie comment: This is for debugging purposes, can be removed later
    print("shape", shape)  # Jackie comment: This is for debugging purposes, can be removed later
    return type_checking_return_actual_dtype(domain, pruned, shape)

# -------------------------------
# Source: https://arxiv.org/pdf/2311.06839.pdf
# -------------------------------
def applyPruningAdaptive(domain):
    """
    Applies adaptive pruning by determining a dynamic prune ratio.
    The prune ratio is set as the maximum value plus a small constant.

    Parameters:
        domain: Input data (list, numpy array, or tensor).

    Returns:
        Adaptively pruned data.
    """
    value, shape = type_checking_and_return_lists(domain)
    pruned = []
    prune_ratio = max(value) + 0.1  # Dynamic prune ratio
    for i in range(len(value)):
        if abs(value[i]) < prune_ratio:
            rnd_tmp = random.random()
            if abs(value[i]) > rnd_tmp * prune_ratio:
                if value[i] > 0:
                    pruned.append(prune_ratio)
                else:
                    pruned.append(-prune_ratio)
            else:
                pruned.append(0)
    print("domain", domain)  # Jackie comment: This is for debugging purposes, can be removed later
    print("pruned", pruned)  # Jackie comment: This is for debugging purposes, can be removed later
    print("shape", shape)  # Jackie comment: This is for debugging purposes, can be removed later
    return type_checking_return_actual_dtype(domain, pruned, shape)

# -------------------------------
# Source: https://arxiv.org/pdf/2311.06839.pdf
# -------------------------------
def applyPruningDP(domain, prune_ratio, sensitivity, epsilon):
    """
    Applies pruning with differential privacy.
    After pruning the values, Laplace noise is added to the pruned values.

    Parameters:
        domain: Input data (list, numpy array, or tensor).
        prune_ratio (float): Pruning threshold.
        sensitivity (float): Sensitivity of the data.
        epsilon (float): Privacy parameter.

    Returns:
        Differentially private pruned data.
    """
    value, shape = type_checking_and_return_lists(domain)
    tmpValue = applyPruning(value, prune_ratio)
    privatized = []
    for i in range(len(tmpValue)):
        privatized.append(tmpValue[i] + np.random.laplace(loc=0, scale=sensitivity / epsilon))
    print("domain", domain)  # Jackie comment: This is for debugging purposes, can be removed later
    print("privatized", privatized)  # Jackie comment: This is for debugging purposes, can be removed later
    print("shape", shape)  # Jackie comment: This is for debugging purposes, can be removed later
    return type_checking_return_actual_dtype(domain, privatized, shape)