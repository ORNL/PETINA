import math
import random
import torch
import numpy as np
from scipy import stats as st

from PETINA.Data_Conversion_Helper import type_checking_and_return_lists, type_checking_return_actual_dtype


#

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
    return type_checking_return_actual_dtype(domain, privatized, shape)

#-----------Jackie work ---------
# -------------------------------
# Source: https://github.com/Samuel-Maddock/pure-LDP/tree/master/pure_ldp/frequency_oracles/apple_cms
# Source: https://machinelearning.apple.com/research/learning-with-privacy-at-scale
# -------------------------------

#Apply Apple CMS (Count-Min Sketch) for differential privacy. Centralized.
def centralized_count_mean_sketch(data, epsilon, k, m, is_hadamard=False):
    hash_funcs = generate_hash_funcs(k, m)
    n = len(data)

    if is_hadamard:
        if not (m & (m - 1)) == 0:
            raise ValueError("m must be a power of 2 for Hadamard sketch")
        had = hadamard(m)
        c = (math.exp(epsilon) + 1) / (math.exp(epsilon) - 1)
        sketch_matrix = np.zeros((k, m))
        for value in data:
            value = str(value)
            j = random.randint(0, k - 1)
            h_j = hash_funcs[j](value)
            w = had[:, h_j]
            l = random.randint(0, m - 1)
            b = random.choices([-1, 1], weights=[1 / (1 + math.exp(epsilon)), 1 - 1 / (1 + math.exp(epsilon))])[0]
            sketch_matrix[j][l] += k * c * b * w[l]
        transformed_matrix = np.matmul(sketch_matrix, had.T)
        estimate_matrix = transformed_matrix
    else:
        c = (math.exp(epsilon / 2) + 1) / (math.exp(epsilon / 2) - 1)
        p = 1 / (1 + math.exp(epsilon / 2))
        sketch_matrix = np.zeros((k, m))
        ones = np.ones(m)
        for value in data:
            value = str(value)
            j = random.randint(0, k - 1)
            h_j = hash_funcs[j](value)
            v = np.full(m, -1)
            v[h_j] = 1
            v[np.random.rand(*v.shape) < p] *= -1
            sketch_matrix[j] += k * ((c / 2) * v + 0.5 * ones)
        estimate_matrix = sketch_matrix

    def estimate(val):
        val = str(val)
        freq_sum = 0
        for i in range(k):
            freq_sum += estimate_matrix[i][hash_funcs[i](val)]
        return (m / (m - 1)) * ((1 / k) * freq_sum - (n / m))

    return estimate
# Helper function to generate hash functions for CMS
def generate_hash_funcs(k, m):
    # In a real scenario, these would be cryptographically secure hash functions
    # For CMS, we need hash functions that map to {0, ..., m-1}
    # For simplicity, we'll use a random seed for each hash function
    hash_funcs = []
    for _ in range(k):
        # Using Python's built-in hash for simplicity, but it's not cryptographically secure
        # For actual LDP, you'd use universal hash families
        seed = random.randint(0, 1000000)
        hash_funcs.append(lambda x, seed=seed, m=m: (hash(str(x) + str(seed))) % m)
    return hash_funcs

#Implementation of CMS as Server and Client seperately.
class Client_PETINA_CMS:
    def __init__(self, k, m, hash_funcs):
        """
        Initializes the client for PETINA Centralized Count-Mean Sketch.
        The client's role is simply to hash its item and prepare it for the server.
        No differential privacy noise is added by the client in this centralized model.

        Args:
            k (int): Number of hash functions.
            m (int): Size of each hash array.
            hash_funcs (list): List of hash functions.
        """
        self.k = k
        self.m = m
        self.hash_funcs = hash_funcs

    def prepare_item(self, item):
        """
        Prepares a single item to be sent to the server.
        It randomly selects one of the k hash functions and computes the hashed index.

        Args:
            item: The data item to be processed.

        Returns:
            tuple: A tuple (j, h_j_x) where j is the index of the chosen hash function
                   and h_j_x is the hashed value of the item using that function.
        """
        value = str(item)
        j = random.randint(0, self.k - 1)  # Client randomly picks one hash function
        h_j_x = self.hash_funcs[j](value) # Client computes the hashed index
        return (j, h_j_x)


class Server_PETINA_CMS:
    def __init__(self, epsilon, k, m, hash_funcs, is_hadamard=False):
        """
        Initializes the server for PETINA Centralized Count-Mean Sketch.
        The server aggregates prepared items from clients and applies the privacy mechanism.

        Args:
            epsilon (float): Privacy parameter epsilon.
            k (int): Number of hash functions.
            m (int): Size of each hash array.
            hash_funcs (list): List of hash functions (must be the same as client's).
            is_hadamard (bool): Whether to use Hadamard transform for noise (if m is a power of 2).
        """
        self.epsilon = epsilon
        self.k = k
        self.m = m
        self.hash_funcs = hash_funcs
        self.is_hadamard = is_hadamard
        self.n_total_items = 0  # To keep track of the total number of items processed

        if self.is_hadamard:
            if not (self.m & (self.m - 1)) == 0:
                raise ValueError("m must be a power of 2 for Hadamard sketch")
            self.had = hadamard(self.m)
            self.c_had = (math.exp(self.epsilon) + 1) / (math.exp(self.epsilon) - 1)
            self.sketch_matrix = np.zeros((self.k, self.m))
        else:
            self.c_non_had = (math.exp(self.epsilon / 2) + 1) / (math.exp(self.epsilon / 2) - 1)
            self.p_non_had = 1 / (1 + math.exp(self.epsilon / 2))
            self.sketch_matrix = np.zeros((self.k, self.m))
            self.ones_vector = np.ones(self.m)

        self.estimate_matrix = None # This will be computed after all data is aggregated

    def aggregate_item(self, prepared_item):
        """
        Aggregates a single prepared item received from a client.
        This is where the sketch is built and the noise is implicitly added based on the
        original centralized logic.

        Args:
            prepared_item (tuple): A tuple (j, h_j_x) from the client.
        """
        self.n_total_items += 1
        j, h_j_x = prepared_item

        if self.is_hadamard:
            w = self.had[:, h_j_x]
            l = random.randint(0, self.m - 1) # Random index for Hadamard noise application
            # This 'b' is the randomized response for Hadamard
            b = random.choices([-1, 1], weights=[1 / (1 + math.exp(self.epsilon)), 1 - 1 / (1 + math.exp(self.epsilon))])[0]
            self.sketch_matrix[j][l] += self.k * self.c_had * b * w[l]
        else:
            # This 'v' is the randomized response vector for non-Hadamard
            v = np.full(self.m, -1)
            v[h_j_x] = 1
            v[np.random.rand(*v.shape) < self.p_non_had] *= -1 # Noise applied here
            self.sketch_matrix[j] += self.k * ((self.c_non_had / 2) * v + 0.5 * self.ones_vector)

    def finalize_sketch(self):
        """
        Finalizes the sketch after all items have been aggregated.
        This performs the final transformation if using Hadamard, or sets the
        estimate matrix directly for non-Hadamard.
        """
        if self.is_hadamard:
            self.estimate_matrix = np.matmul(self.sketch_matrix, self.had.T)
        else:
            self.estimate_matrix = self.sketch_matrix

    def estimate(self, val):
        """
        Estimates the frequency of a given value from the finalized sketch.

        Args:
            val: The value for which to estimate the frequency.

        Returns:
            float: The estimated frequency of the value.
        """
        if self.estimate_matrix is None:
            raise RuntimeError("Sketch not finalized. Call finalize_sketch() first.")

        val = str(val)
        freq_sum = 0
        for i in range(self.k):
            freq_sum += self.estimate_matrix[i][self.hash_funcs[i](val)]
        return (self.m / (self.m - 1)) * ((1 / self.k) * freq_sum - (self.n_total_items / self.m))


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