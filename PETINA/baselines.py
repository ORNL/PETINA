import math
import random
import torch
import numpy as np
from scipy import stats as st

# This module provides a collection of functions for data processing, differential privacy mechanisms,
# encoding, perturbation, clipping, and pruning. It supports data types such as Python lists, NumPy arrays, and PyTorch tensors.

# -------------------------------
# Utility functions for type conversion
# -------------------------------

def type_checking_and_return_lists(domain):
    """
    Converts the input data (tensor, numpy array, or list) to a list and returns its shape (if applicable).

    Parameters:
        domain: Input data (torch.Tensor, np.ndarray, or list)

    Returns:
        items: A list representation of the input data.
        shape: Original shape information (for tensors and numpy arrays; 0 for lists).
    """
    if isinstance(domain, torch.Tensor):
        items, shape = torch_to_list(domain)  # Convert torch tensor to list
    elif isinstance(domain, np.ndarray):
        items, shape = numpy_to_list(domain)  # Convert numpy array to list
    elif isinstance(domain, list):
        items = domain
        shape = 0  # Shape information is not used for plain lists
    else:
        raise ValueError("only takes list, ndarray, tensor type")
    
    return items, shape

def type_checking_return_actual_dtype(domain, result, shape):
    """
    Converts a processed list back to the original data type of 'domain'.

    Parameters:
        domain: The original input data (to check its type).
        result: The processed data as a list.
        shape: The shape information for conversion (if applicable).

    Returns:
        The result converted back to the original data type.
    """
    if isinstance(domain, torch.Tensor):
        return list_to_torch(result, shape)  # Convert list back to torch tensor
    elif isinstance(domain, np.ndarray):
        return list_to_numpy(result, shape)  # Convert list back to numpy array
    else:  # If input was a list, return the list as is
         return result

# -------------------------------
# Differential Privacy Mechanisms
# -------------------------------

# Depending on your data, parameters for each privacy technique below will need to be changed. The default 
# parameter might not be the best value and can affect the accuracy of your model
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

# -------------------------------
# Encoding and Perturbation Functions
# -------------------------------

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

# -------------------------------
# Parameter Calculation Helpers
# -------------------------------

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

# -------------------------------
# Sparse Vector Technique (SVT)
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

# -------------------------------
# Additional Perturbation and Aggregation Methods
# -------------------------------

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

# -------------------------------
# Histogram Encoding Methods
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
# Clipping Functions
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

# -------------------------------
# Pruning Functions
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
    return type_checking_return_actual_dtype(domain, pruned, shape)

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
    return type_checking_return_actual_dtype(domain, pruned, shape)

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

    return type_checking_return_actual_dtype(domain, privatized, shape)

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
# Miscellaneous Helper Functions
# -------------------------------

def shuffle(a):
    """
    Shuffles a list of lists in place.

    Example:
        a = [[1,2,3], [4,5,6], [7,8,9]]
        result = shuffle(a) 
        # The list 'a' will be randomly rearranged.
    """
    random.shuffle(a)
    return a

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

# -------------------------------
# Data Conversion Helper Functions
# -------------------------------

def numpy_to_list(nd_array):
    """
    Converts a NumPy array to a flattened list and returns its original shape.

    Parameters:
        nd_array (np.ndarray): Input NumPy array.

    Returns:
        A tuple (flattened_list, original_shape).
    """
    flattened_list = nd_array.flatten().tolist()
    nd_array_shape = nd_array.shape
    return flattened_list, nd_array_shape

def list_to_numpy(flattened_list, nd_array_shape):
    """
    Converts a flattened list back to a NumPy array with the given shape.

    Parameters:
        flattened_list (list): Flattened list of values.
        nd_array_shape (tuple): Desired shape for the NumPy array.

    Returns:
        A NumPy array with the specified shape.
    """
    reverted_ndarray = np.array(flattened_list).reshape(nd_array_shape)
    return reverted_ndarray

def torch_to_list(torch_tensor):
    """
    Converts a PyTorch tensor to a flattened list and returns its original shape.

    Parameters:
        torch_tensor (torch.Tensor): Input tensor.

    Returns:
        A tuple (flattened_list, original_shape).
    """
    flattened_list = torch_tensor.flatten().tolist()
    tensor_shape = torch_tensor.shape
    return flattened_list, tensor_shape

def list_to_torch(flattened_list, tensor_shape):
    """
    Converts a flattened list back to a PyTorch tensor with the given shape.

    Parameters:
        flattened_list (list): Flattened list of values.
        tensor_shape (tuple): Desired shape for the tensor.

    Returns:
        A PyTorch tensor with the specified shape.
    """
    reverted_tensor = torch.as_tensor(flattened_list).reshape(tensor_shape)
    return reverted_tensor
