import math
import random
import torch
import numpy as np
from scipy import stats as st

from PETINA.Data_Conversion_Helper import type_checking_and_return_lists, type_checking_return_actual_dtype

# -------------------------------
# Encoding and Perturbation Functions
# Source: Úlfar Erlingsson, Vasyl Pihur, and Aleksandra Korolova. Rappor: randomized aggregatable privacy-preserving ordinal response.
# In Proceedings of the 2014 ACM SIGSAC Conference on Computer and Communications Security, CCS '14, 1054–1067. New York, NY, USA, 2014.
# Association for Computing Machinery. URL: https://doi.org/10.1145/2660267.2660348, doi:10.1145/2660267.2660348.
# and
# Source: The Algorithmic Foundations of Differential Privacy by Cynthia Dwork and Aaron Roth. Foundations and Trends in Theoretical Computer Science.
# -------------------------------

def perturb_bit(bit, p, q):
    """
    Perturbs a single bit using random response.

    Parameters:
        bit (int): The binary bit (0 or 1).
        p (float): Probability of keeping 1 as 1.
        q (float): Probability of flipping 0 to 1.

    Returns:
        The perturbed bit.
    """
    sample = np.random.random()  # Generate a random float in [0, 1)
    if bit == 1:
        return 1 if sample <= p else 0
    elif bit == 0:
        return 1 if sample <= q else 0


def perturb(encoded_response, p, q):
    """
    Applies perturbation to an entire encoded response vector.

    Parameters:
        encoded_response (list): A list of binary bits.
        p, q (float): Perturbation probabilities.

    Returns:
        A perturbed version of the encoded response.
    """
    return [perturb_bit(b, p, q) for b in encoded_response]


def get_q(p, eps):
    """
    Computes q given p and epsilon based on the relation:
    p(1-q) / q(1-p) = exp(eps)

    Parameters:
        p (float): Probability of keeping a bit.
        eps (float): Privacy parameter.

    Returns:
        q (float): Computed probability.
    """
    qinv = 1 + (math.exp(eps) * (1.0 - p) / p)
    q = 1.0 / qinv
    return q


def get_gamma_sigma(p, eps):
    """
    Computes gamma and sigma parameters for the Gaussian mechanism.

    Parameters:
        p (float): Probability parameter.
        eps (float): Privacy parameter.

    Returns:
        gamma (float): Threshold value derived from the inverse survival function.
        sigma (float): Noise scaling factor.
    """
    qinv = 1 + (math.exp(eps) * (1.0 - p) / p)
    q = 1.0 / qinv
    gamma = st.norm.isf(q)  # Inverse survival function of standard normal
    # Compute conditional expectation adjustments.
    unnorm_mu = st.norm.pdf(gamma) * (-(1.0 - p) / st.norm.cdf(gamma) + p / st.norm.sf(gamma))
    sigma = 1.0 / unnorm_mu
    return gamma, sigma


def get_p(eps, return_sigma=False):
    """
    Determines the optimal probability p for a given epsilon by searching a range
    and selecting the one with minimum sigma (noise scale).

    Parameters:
        eps (float): Privacy parameter.
        return_sigma (bool): If True, also return the corresponding sigma.

    Returns:
        Optimal p value (and sigma if return_sigma is True).
    """
    plist = np.arange(0.01, 1.0, 0.01)
    glist = []
    slist = []
    for p in plist:
        gamma, sigma = get_gamma_sigma(p, eps)
        glist.append(gamma)
        slist.append(sigma)
    ii = np.argmin(slist)
    if return_sigma:
        return plist[ii], slist[ii]
    else:
        return plist[ii]


def aggregate(responses, p=0.75, q=0.25):
    """
    Aggregates a list of perturbed responses to estimate the original counts.

    Parameters:
        responses (list of lists): Perturbed one-hot encoded responses.
        p (float): Probability parameter used during perturbation.
        q (float): Secondary probability parameter used during perturbation.

    Returns:
        A list of estimated counts for each element in the domain.
    """
    sums = np.sum(responses, axis=0)  # Sum across all responses
    n = len(responses)
    # Adjust the sums to compensate for the random response mechanism.
    return [(v - n * q) / (p - q) for v in sums]


def the_aggregation_and_estimation(answers, epsilon=0.1, theta=1.0):
    """
    Aggregates the perturbed answers and estimates the original counts.

    Parameters:
        answers (list of lists): Perturbed responses.
        epsilon (float): Privacy parameter.
        theta (float): Threshold parameter.

    Returns:
        A list of estimated counts as integers.
    """
    # Compute the probabilities based on epsilon and theta.
    p = 1 - 0.5 * math.exp(epsilon / 2 * (1.0 - theta))
    q = 0.5 * math.exp(epsilon / 2 * (0.0 - theta))

    sums = np.sum(answers, axis=0)
    n = len(answers)

    # Adjust the sums to recover the original counts.
    return [int((i - n * q) / (p - q)) for i in sums]


def she_perturb_bit(bit, epsilon=0.1):
    """
    Perturbs a single bit using Laplace noise.

    Parameters:
        bit (float/int): The bit value.
        epsilon (float): Privacy parameter.

    Returns:
        Perturbed bit.
    """
    return bit + np.random.laplace(loc=0, scale=2 / epsilon)


def she_perturbation(encoded_response, epsilon=0.1):
    """
    Applies Laplace noise to each element of an encoded response.

    Parameters:
        encoded_response (list): A list of bits.
        epsilon (float): Privacy parameter.

    Returns:
        List of perturbed bits.
    """
    return [she_perturb_bit(b, epsilon) for b in encoded_response]


def the_perturb_bit(bit, epsilon=0.1, theta=1.0):
    """
    Perturbs a single bit, thresholds the result, and returns either 1.0 or 0.0.

    Parameters:
        bit (float/int): The bit value.
        epsilon (float): Privacy parameter.
        theta (float): Threshold parameter.

    Returns:
        1.0 if the perturbed value exceeds theta, otherwise 0.0.
    """
    val = bit + np.random.laplace(loc=0, scale=2 / epsilon)
    return 1.0 if val > theta else 0.0


def the_perturbation(encoded_response, epsilon=0.1, theta=1.0):
    """
    Applies the threshold-based perturbation to an encoded response.

    Parameters:
        encoded_response (list): A list of bits.
        epsilon (float): Privacy parameter.
        theta (float): Threshold value.

    Returns:
        List of perturbed bits (either 0.0 or 1.0).
    """
    return [the_perturb_bit(b, epsilon, theta) for b in encoded_response]


def encode(response, domain):
    """
    Encodes a response into a one-hot representation with respect to the domain.

    Parameters:
        response: The value to encode.
        domain: The set of possible values.

    Returns:
        A list with 1 where the domain element equals the response, else 0.
    """
    return [1 if d == response else 0 for d in domain]


def unary_epsilon(p, q):
    """
    Computes the effective epsilon for unary encoding based on probabilities p and q.

    Parameters:
        p (float): Probability of preserving a bit.
        q (float): Probability of flipping a bit to 1.

    Returns:
        The computed epsilon value.
    """
    return np.log((p * (1 - q)) / ((1 - p) * q))


# -------------------------------
# Encoding Methods
# -------------------------------

# -------------------------------
# Source: https://livebook.manning.com/book/privacy-preserving-machine-learning/chapter-4/v-4/103
# -------------------------------
def histogramEncoding(value):
    """
    Implements histogram encoding with differential privacy using Laplace perturbation.

    Parameters:
        value: Input data (list, numpy array, or tensor).

    Returns:
        Privatized counts corresponding to the input data.
    """
    domain, shape = type_checking_and_return_lists(value)

    # Perturb the one-hot encoded responses for each element.
    responses = [she_perturbation(encode(r, domain)) for r in domain]
    counts = aggregate(responses)
    t = list(zip(domain, counts))

    privatized = [count for _, count in t]
    return type_checking_return_actual_dtype(value, privatized, shape)


# -------------------------------
# Source: https://livebook.manning.com/book/privacy-preserving-machine-learning/chapter-4/v-4/103
# -------------------------------
def histogramEncoding_t(value):
    """
    An alternative histogram encoding using threshold-based perturbation and aggregation.

    Parameters:
        value: Input data (list, numpy array, or tensor).

    Returns:
        Estimated counts derived from the perturbed responses.
    """
    domain, shape = type_checking_and_return_lists(value)
    # Apply threshold-based perturbation to the one-hot encoding.
    the_perturbed_answers = [the_perturbation(encode(r, domain)) for r in domain]
    # Estimate the original counts.
    estimated_answers = the_aggregation_and_estimation(the_perturbed_answers)
    return type_checking_return_actual_dtype(value, estimated_answers, shape)


# -------------------------------
# Source: Úlfar Erlingsson, Vasyl Pihur, and Aleksandra Korolova. Rappor: randomized aggregatable privacy-preserving ordinal response.
# In Proceedings of the 2014 ACM SIGSAC Conference on Computer and Communications Security, CCS '14, 1054–1067. New York, NY, USA, 2014.
# Association for Computing Machinery. URL: https://doi.org/10.1145/2660267.2660348, doi:10.1145/2660267.2660348.
# -------------------------------
def unaryEncoding(value, p=0.75, q=0.25):
    """
    Applies unary encoding with differential privacy.
    Each value is encoded as a one-hot vector, perturbed, and then aggregated.

    Parameters:
        value: Input data (list, numpy array, or tensor).
        p (float): Probability of keeping an encoded bit unchanged.
        q (float): Probability of flipping an encoded bit to 1 when it is 0.

    Returns:
        A list of tuples pairing each unique value with its privatized count.
    """
    # Convert input data to list.
    domain, _ = type_checking_and_return_lists(value)
    # Get unique values in the domain.
    unique_domain = list(set(domain))

    # For each value, encode and perturb it.
    responses = [perturb(encode(r, unique_domain), p, q) for r in domain]
    # Aggregate perturbed responses.
    counts = aggregate(responses, p, q)
    # Zip unique values with their estimated counts.
    t = list(zip(unique_domain, counts))
    return t
