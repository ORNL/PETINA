import math
import random
import torch
import numpy as np
from scipy import stats as st

from PETINA.Data_Conversion_Helper import type_checking_and_return_lists, type_checking_return_actual_dtype


# -------------------------------
# Source: The Algorithmic Foundations of Differential Privacy by Cynthia Dwork and Aaron Roth. Foundations and Trends in Theoretical Computer Science.
# Vol. 9, no. 3–4, pp. 211‐407, Aug. 2014. doi:10.1561/0400000042
# -------------------------------
def applyFlipCoin(probability, domain):
    """
    Applies a "flip coin" mechanism to each item in the input domain.
    For each item, with a probability 'probability', the original item is kept.
    Otherwise, a random integer between the minimum and maximum of the list is used.

    Parameters:
        probability (float): Probability (between 0 and 1) to keep the original item.
        domain: Input data (list, numpy array, or tensor).

    Returns:
        Data with each item either preserved or replaced with a random value,
        in the same format as the input.
    """
    # Ensure the probability is valid.
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")

    # Convert input data to list.
    items, shape = type_checking_and_return_lists(domain)

    # Create a list of boolean values; True with probability 'probability'
    prob = [np.random.rand() < probability for _ in items]

    result = []
    # Determine the minimum and maximum values in the list for random replacement.
    item_min = min(items)
    item_max = max(items)

    # For each item, decide whether to keep it or replace it with a random value.
    for p, n in zip(prob, items):
        if p == True:
            result.append(n)  # Keep the original value
        else:
            result.append(random.randint(item_min, item_max))  # Replace with random integer

    # Convert the result back to the original data type.

    return type_checking_return_actual_dtype(domain, result, shape)


# -------------------------------
# Source: Differential Privacy by Cynthia Dwork, International Colloquium on Automata, Languages and Programming (ICALP) 2006, p. 1–12. doi:10.1007/11787006_1
# -------------------------------
def applyDPGaussian(domain, delta=10e-5, epsilon=1, gamma=1):
    """
    Applies Gaussian noise to the input data for differential privacy.

    Parameters:
        domain: Input data (list, numpy array, or tensor).
        delta (float): Failure probability (default: 1e-5).
        epsilon (float): Privacy parameter (default: 1.0).
        gamma (float): Scaling factor for noise (default: 1).

    Returns:
        Data with added Gaussian noise in the same format as the input.
    """
    data, shape = type_checking_and_return_lists(domain)

    # Calculate the standard deviation for the Gaussian noise.
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * gamma / epsilon
    # Add Gaussian noise to each data point.
    privatized = data + np.random.normal(loc=0, scale=sigma, size=len(data))
    return type_checking_return_actual_dtype(domain, privatized, shape)


# -------------------------------
# Source: Ilya Mironov. Renyi differential privacy. In Computer Security Foundations Symposium (CSF), 2017 IEEE 30th, 263–275. IEEE, 2017.
# -------------------------------
def applyRDPGaussian(domain, sensitivity=1, alpha=10, epsilon_bar=1):
    """
    Applies Gaussian noise using the Rényi Differential Privacy (RDP) mechanism.

    Parameters:
        domain: Input data (list, numpy array, or tensor).
        sensitivity (float): Sensitivity of the data (default: 1).
        alpha (float): RDP parameter (default: 10).
        epsilon_bar (float): Privacy parameter (default: 1).

    Returns:
        Data with added Gaussian noise.
    """
    data, shape = type_checking_and_return_lists(domain)
    # Calculate sigma based on sensitivity, alpha, and epsilon_bar.
    sigma = np.sqrt((sensitivity**2 * alpha) / (2 * epsilon_bar))
    # Add Gaussian noise for each element.
    privatized = [v + np.random.normal(loc=0, scale=sigma) for v in data]

    return type_checking_return_actual_dtype(domain, privatized, shape)


# -------------------------------
# Source: Mark Bun and Thomas Steinke. Concentrated differential privacy: simplifications, extensions, and lower bounds. In Theory of Cryptography Conference, 635–658. Springer, 2016.
# -------------------------------
def applyDPExponential(domain, sensitivity=1, epsilon=1, gamma=1.0):
    """
    Applies exponential noise to the input data for differential privacy.

    Parameters:
        domain: Input data (list, numpy array, or tensor).
        sensitivity: Maximum change by a single individual's data (default: 1).
        epsilon: Privacy parameter (default: 1).
        gamma: Scaling factor for noise (default: 1.0).

    Returns:
        Data with added exponential noise in the same format as the input.
    """
    data, shape = type_checking_and_return_lists(domain)

    # Determine the scale for the exponential distribution.
    scale = sensitivity * gamma / epsilon

    # Generate exponential noise and randomly flip its sign to create a symmetric noise distribution.
    noise = np.random.exponential(scale, size=len(data))
    signs = np.random.choice([-1, 1], size=len(data))
    noise = noise * signs

    # Add the noise to the original data.
    privatized = np.array(data) + noise

    # Convert the result back to a list.
    privatized = privatized.tolist()
    return type_checking_return_actual_dtype(domain, privatized, shape)


# -------------------------------
# Source: Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis.
# In Proceedings of the Third Conference on Theory of Cryptography, TCC'06, 265–284. Berlin, Heidelberg, 2006. Springer-Verlag.
# URL: https://doi.org/10.1007/11681878_14, doi:10.1007/11681878_14.
# -------------------------------
def applyDPLaplace(domain, sensitivity=1, epsilon=1, gamma=1):
    """
    Applies Laplace noise to the input data for differential privacy.

    Parameters:
        domain: Input data (list, numpy array, or tensor).
        sensitivity: Maximum change by a single individual's data (default: 1).
        epsilon: Privacy parameter (default: 1).
        gamma: Scaling factor for noise (default: 1).

    Returns:
        Data with added Laplace noise in the same format as the input.
    """
    data, shape = type_checking_and_return_lists(domain)
    # Add Laplace noise to each element.
    privatized = data + np.random.laplace(loc=0, scale=sensitivity * gamma / epsilon, size=len(data))
    return type_checking_return_actual_dtype(domain, privatized, shape)