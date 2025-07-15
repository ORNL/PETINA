import math
import random
import torch
import numpy as np
from scipy import stats as st
from PETINA.Data_Conversion_Helper import type_checking_and_return_lists, type_checking_return_actual_dtype
from PETINA.package.csvec.csvec import CSVec

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


# -------------------------------
# Source: Differential Privacy by Cynthia Dwork, International Colloquium on Automata, Languages and Programming (ICALP) 2006, p. 1–12. doi:10.1007/11787006_1
# -------------------------------

# def applyDPGaussian(domain, delta=1e-5, epsilon=1.0, gamma=1.0):
#     """
#     Applies Gaussian noise to the input data for differential privacy.

#     Parameters:
#         domain: Input data (list, numpy array, or tensor).
#         delta (float): Failure probability (default: 1e-5).
#         epsilon (float): Privacy parameter (default: 1.0).
#         gamma (float): Scaling factor for noise (default: 1.0).

#     Returns:
#         Data with added Gaussian noise in the same format as the input.
#     """
#     data, shape = type_checking_and_return_lists(domain)

#     sigma = np.sqrt(2 * np.log(1.25 / delta)) * gamma / epsilon
#     noise = np.random.normal(loc=0, scale=sigma, size=len(data))
#     privatized = np.array(data) + noise

#     return type_checking_return_actual_dtype(domain, privatized.tolist(), shape)

def applyDPGaussian(domain, delta=1e-5, epsilon=0.1, gamma=1.0, accountant=None):
    """
    Applies Gaussian noise to the input data for differential privacy,
    and optionally tracks budget via a BudgetAccountant.

    Parameters:
        domain: Input data (list, numpy array, or tensor).
        delta (float): Failure probability (default: 1e-5).
        epsilon (float): Privacy parameter (default: 1.0).
        gamma (float): Scaling factor for noise (default: 1.0).
        accountant (BudgetAccountant, optional): Tracks spend for (ε, δ).

    Returns:
        Data with added Gaussian noise in the same format as the input.
    """
    # Flatten to list
    data, shape = type_checking_and_return_lists(domain)

    # Compute σ for (ε, δ)-Gaussian mechanism
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * gamma / epsilon

    # Add Gaussian noise
    privatized = np.array(data) + np.random.normal(loc=0, scale=sigma, size=len(data))

    # Budget accounting
    if accountant is not None:
        # Spend the exact (ε, δ) for this invocation
        accountant.spend(epsilon, delta)
        # (Optional) debug print
        print(f"Gaussian: spent ε={epsilon}, δ={delta}; remaining={accountant.remaining()}")

    # Restore to original type/shape
    return type_checking_return_actual_dtype(domain, privatized, shape)

# -------------------------------
# Source: Ilya Mironov. Renyi differential privacy. In Computer Security Foundations Symposium (CSF), 2017 IEEE 30th, 263–275. IEEE, 2017.
# -------------------------------

def applyRDPGaussian(domain, sensitivity=1.0, alpha=10.0, epsilon_bar=1.0):
    """
    Applies Gaussian noise using the Rényi Differential Privacy (RDP) mechanism.

    Parameters:
        domain: Input data (list, numpy array, or tensor).
        sensitivity (float): Sensitivity of the data (default: 1.0).
        alpha (float): RDP parameter (default: 10.0).
        epsilon_bar (float): Privacy parameter (default: 1.0).

    Returns:
        Data with added Gaussian noise.
    """
    data, shape = type_checking_and_return_lists(domain)
    
    sigma = np.sqrt((sensitivity ** 2 * alpha) / (2 * epsilon_bar))
    noise = np.random.normal(loc=0, scale=sigma, size=len(data))
    privatized = np.array(data) + noise

    return type_checking_return_actual_dtype(domain, privatized.tolist(), shape)



# -------------------------------
# Source: Mark Bun and Thomas Steinke. Concentrated differential privacy: simplifications, extensions, and lower bounds. In Theory of Cryptography Conference, 635–658. Springer, 2016.
# -------------------------------

def applyDPExponential(domain, sensitivity=1.0, epsilon=1.0, gamma=1.0):
    """
    Applies exponential noise to the input data for differential privacy.

    Parameters:
        domain: Input data (list, numpy array, or tensor).
        sensitivity (float): Maximum change by a single individual's data (default: 1.0).
        epsilon (float): Privacy parameter (default: 1.0).
        gamma (float): Scaling factor for noise (default: 1.0).

    Returns:
        Data with added exponential noise in the same format as the input.
    """
    data, shape = type_checking_and_return_lists(domain)

    scale = sensitivity * gamma / epsilon

    # Generate symmetric exponential noise by sampling exponential and randomly flipping signs
    noise = np.random.exponential(scale=scale, size=len(data))
    signs = np.random.choice([-1, 1], size=len(data))
    noise *= signs

    privatized = np.array(data) + noise

    return type_checking_return_actual_dtype(domain, privatized.tolist(), shape)


# -------------------------------
# Source: Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis.
# In Proceedings of the Third Conference on Theory of Cryptography, TCC'06, 265–284. Berlin, Heidelberg, 2006. Springer-Verlag.
# URL: https://doi.org/10.1007/11681878_14, doi:10.1007/11681878_14.
# -------------------------------

# def applyDPLaplace(domain, sensitivity=1.0, epsilon=1.0, gamma=1.0):
#     """
#     Applies Laplace noise to the input data for differential privacy.

#     Parameters:
#         domain: Input data (list, numpy array, or tensor).
#         sensitivity (float): Maximum change by a single individual's data (default: 1.0).
#         epsilon (float): Privacy parameter (default: 1.0).
#         gamma (float): Scaling factor for noise (default: 1.0).

#     Returns:
#         Data with added Laplace noise in the same format as the input.
#     """
#     data, shape = type_checking_and_return_lists(domain)
#     scale = sensitivity * gamma / epsilon
#     noise = np.random.laplace(loc=0, scale=scale, size=len(data))
#     privatized = data + noise
#     return type_checking_return_actual_dtype(domain, privatized, shape)
def applyDPLaplace(domain, sensitivity=1, epsilon=0.01, gamma=1, accountant=None):
    """
    Applies Laplace noise to the input data for differential privacy.
    Modified to track budget with a BudgetAccountant.

    Parameters:
        domain: Input data (list, numpy array, or tensor).
        sensitivity: Maximum change by a single individual's data (default: 1).
        epsilon: Privacy parameter (default: 1).
        gamma: Scaling factor for noise (default: 1).
        accountant (BudgetAccountant, optional): The budget accountant to track spend.

    Returns:
        Data with added Laplace noise in the same format as the input.
    """
    # Use helper functions to convert input data to a flattened list and get its shape
    data, shape = type_checking_and_return_lists(domain)
    
    # Calculate the scale for the Laplace distribution.
    # This maintains the original noise calculation from the PETINA function.
    scale = sensitivity * gamma / epsilon
    
    # Add Laplace noise to each element of the flattened data.
    privatized = np.array(data) + np.random.laplace(loc=0, scale=scale, size=len(data))

    # --- Inject the budget tracking logic here ---
    if accountant is not None:
        print("Accountant is present, spending budget for Laplace noise addition.")
        
        # The budget cost is based on the `epsilon` parameter provided to the function.
        # For Laplace noise, the delta cost is 0.
        cost_epsilon, cost_delta = epsilon, 0.0
        
        # The `spend` method will internally check if the budget is exceeded.
        accountant.spend(cost_epsilon, cost_delta)
        
        # Print the total spent budget for debugging and monitoring purposes.
        print("Total spend: %r" % (accountant.total(),))
        
    # Convert the processed flattened list back to the original data type and shape.
    return type_checking_return_actual_dtype(domain, privatized, shape)



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
    for v in value:
        if abs(v) < prune_ratio:
            rnd_tmp = random.random()
            if abs(v) > rnd_tmp * prune_ratio:
                pruned.append(prune_ratio if v > 0 else -prune_ratio)
            else:
                pruned.append(0)
        else:
            pruned.append(v)  # Keep original if not below prune_ratio
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
    for v in value:
        if abs(v) < prune_ratio:
            rnd_tmp = random.random()
            if abs(v) > rnd_tmp * prune_ratio:
                pruned.append(prune_ratio if v > 0 else -prune_ratio)
            else:
                pruned.append(0)
        else:
            pruned.append(v)  # Keep original if not below prune_ratio
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
    pruned_values = applyPruning(value, prune_ratio)
    noise_scale = sensitivity / epsilon
    # Add Laplace noise vectorized for efficiency
    noise = np.random.laplace(loc=0, scale=noise_scale, size=len(pruned_values))
    privatized = np.array(pruned_values) + noise
    return type_checking_return_actual_dtype(domain, privatized.tolist(), shape)


#-----------Jackie work ---------

# -------------------------------
# Source: https://github.com/nikitaivkin/csh#
# -------------------------------
### Count Sketch for Private Aggregation

# Here is the new function, `applyCountSketch`, which uses the `CSVec` library to sketch and un-sketch your input data. This method is valuable for private aggregation in distributed settings like federated learning because it allows you to represent large, high-dimensional vectors (like model updates) with a much smaller sketch while still maintaining a strong estimate of the original data.

def applyCountSketch(domain, num_rows, num_cols):
    """
    Applies the Count Sketch mechanism to the input data.
    The input vector is sketched and then un-sketched to demonstrate
    the approximation capability of the data structure.

    Parameters:
        domain: Input data (list, numpy array, or tensor).
        num_rows (int): The number of rows in the sketch matrix.
        num_cols (int): The number of columns (buckets) in the sketch matrix.

    Returns:
        The reconstructed data after sketching, in the same format as the input.
    """
    # Convert input data to a list for processing.
    items, shape = type_checking_and_return_lists(domain)
    
    # Get the dimension (length) of the flattened vector.
    dimension = len(items)
    
    # Create a CSVec (Count Sketch Vector) object.
    # The dimension is the size of the original vector, and num_rows/num_cols
    # define the sketch matrix size.
    cs_vec = CSVec(d=dimension, c=num_cols, r=num_rows)
    
    # Accumulate the vector into the sketch. This is the sketching step.
    cs_vec.accumulateVec(torch.tensor(items, dtype=torch.float32))
    
    # Un-sketch the vector to get the approximation.
    # This reconstructs the original vector from the sketch.
    unsketched_tensor = cs_vec.unSketch(k=dimension)
    
    # Convert the unsketched tensor back to a list.
    privatized = unsketched_tensor.tolist()
    
    print("domain", domain)
    print("privatized", privatized)
    print("shape", shape)
    
    # Convert the processed list back to the original data type.
    return type_checking_return_actual_dtype(domain, privatized, shape)