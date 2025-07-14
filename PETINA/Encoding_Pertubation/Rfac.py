import math
import numpy as np
import torch
from scipy import stats as st
from PETINA.Data_Conversion_Helper import type_checking_and_return_lists, type_checking_return_actual_dtype

# -------------------------------
# Encoding & Perturbation Functions
# Sources:
# - RAPPOR: Erlingsson et al., CCS '14 (https://doi.org/10.1145/2660267.2660348)
# - Algorithmic Foundations of Differential Privacy (Dwork & Roth)
# -------------------------------

def perturb_bit(bit, p, q):
    """
    Randomized response perturbation for a single bit.

    Args:
        bit (int): Original bit (0 or 1).
        p (float): Probability of keeping bit 1 as 1.
        q (float): Probability of flipping bit 0 to 1.

    Returns:
        int: Perturbed bit.
    """
    sample = np.random.random()
    return 1 if (bit == 1 and sample <= p) or (bit == 0 and sample <= q) else 0


def perturb(encoded_response, p, q):
    """Apply perturbation to an encoded bit vector."""
    return [perturb_bit(b, p, q) for b in encoded_response]


def get_q(p, eps):
    """
    Compute q given p and epsilon using p(1-q)/q(1-p) = exp(eps).

    Returns:
        float: Computed q.
    """
    return 1 / (1 + (math.exp(eps) * (1 - p) / p))


def get_gamma_sigma(p, eps):
    """
    Compute gamma (threshold) and sigma (noise scale) parameters for Gaussian mechanism.

    Returns:
        tuple: (gamma, sigma)
    """
    q = get_q(p, eps)
    gamma = st.norm.isf(q)
    unnorm_mu = st.norm.pdf(gamma) * (-(1 - p) / st.norm.cdf(gamma) + p / st.norm.sf(gamma))
    sigma = 1 / unnorm_mu
    return gamma, sigma


def get_p(eps, return_sigma=False):
    """
    Find optimal p in [0.01, 1) minimizing sigma for given epsilon.

    Returns:
        float or (float, float): p (and sigma if return_sigma=True).
    """
    plist = np.arange(0.01, 1.0, 0.01)
    sigmas = [get_gamma_sigma(p, eps)[1] for p in plist]
    idx = np.argmin(sigmas)
    return (plist[idx], sigmas[idx]) if return_sigma else plist[idx]


def aggregate(responses, p=0.75, q=0.25):
    """
    Aggregate perturbed responses to estimate counts.

    Args:
        responses (list of lists): Perturbed one-hot vectors.
        p (float), q (float): Perturbation probabilities.

    Returns:
        list: Estimated counts per domain element.
    """
    sums = np.sum(responses, axis=0)
    n = len(responses)
    return [(v - n * q) / (p - q) for v in sums]


def the_aggregation_and_estimation(answers, epsilon=0.1, theta=1.0):
    """
    Threshold-based aggregation and count estimation.

    Args:
        answers (list of lists): Perturbed responses.
        epsilon (float): Privacy parameter.
        theta (float): Threshold.

    Returns:
        list: Estimated counts as integers.
    """
    p = 1 - 0.5 * math.exp(epsilon / 2 * (1 - theta))
    q = 0.5 * math.exp(epsilon / 2 * (0 - theta))
    sums = np.sum(answers, axis=0)
    n = len(answers)
    return [int((v - n * q) / (p - q)) for v in sums]


def she_perturb_bit(bit, epsilon=0.1):
    """Perturb a bit using Laplace noise."""
    return bit + np.random.laplace(loc=0, scale=2 / epsilon)


def she_perturbation(encoded_response, epsilon=0.1):
    """Apply Laplace perturbation to each bit."""
    return [she_perturb_bit(b, epsilon) for b in encoded_response]


def the_perturb_bit(bit, epsilon=0.1, theta=1.0):
    """
    Perturb bit with Laplace noise and threshold.

    Returns:
        float: 1.0 if perturbed bit > theta else 0.0.
    """
    val = bit + np.random.laplace(loc=0, scale=2 / epsilon)
    return 1.0 if val > theta else 0.0


def the_perturbation(encoded_response, epsilon=0.1, theta=1.0):
    """Apply threshold perturbation to encoded response."""
    return [the_perturb_bit(b, epsilon, theta) for b in encoded_response]


def encode(response, domain):
    """
    One-hot encode a response relative to domain.

    Args:
        response: Value to encode.
        domain (list): Domain of possible values.

    Returns:
        list: One-hot encoded vector.
    """
    return [1 if d == response else 0 for d in domain]


def unary_epsilon(p, q):
    """Calculate unary encoding privacy parameter epsilon."""
    return np.log((p * (1 - q)) / ((1 - p) * q))


# -------------------------------
# Encoding Methods
# -------------------------------

def histogramEncoding(value):
    """
    Histogram encoding with Laplace perturbation.

    Args:
        value: Input data (list, ndarray, or tensor).

    Returns:
        Perturbed counts matching input format.
    """
    domain, shape = type_checking_and_return_lists(value)
    responses = [she_perturbation(encode(r, domain)) for r in domain]
    counts = aggregate(responses)
    privatized = [count for _, count in zip(domain, counts)]
    return type_checking_return_actual_dtype(value, privatized, shape)


def histogramEncoding_t(value):
    """
    Histogram encoding using threshold perturbation and estimation.

    Args:
        value: Input data (list, ndarray, or tensor).

    Returns:
        Estimated counts matching input format.
    """
    domain, shape = type_checking_and_return_lists(value)
    perturbed_answers = [the_perturbation(encode(r, domain)) for r in domain]
    estimated = the_aggregation_and_estimation(perturbed_answers)
    return type_checking_return_actual_dtype(value, estimated, shape)


def unaryEncoding(value, p=0.75, q=0.25):
    """
    Unary encoding with randomized response perturbation.

    Args:
        value: Input data (list, ndarray, or tensor).
        p (float): Probability to keep 1 as 1.
        q (float): Probability to flip 0 to 1.

    Returns:
        list of (value, estimated count) tuples.
    """
    domain, _ = type_checking_and_return_lists(value)
    unique_domain = list(set(domain))
    responses = [perturb(encode(r, unique_domain), p, q) for r in domain]
    counts = aggregate(responses, p, q)
    return list(zip(unique_domain, counts))
