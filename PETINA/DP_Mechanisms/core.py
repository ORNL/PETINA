import math
import random
import torch
import numpy as np
from scipy import stats as st
from PETINA.Data_Conversion_Helper import TypeConverter
from PETINA.package.csvec.csvec import CSVec

# Depending on your data, parameters for each privacy technique below will need to be changed. The default 
# parameter might not be the best value and can affect the accuracy of your model

def applyFlipCoin(probability, domain):
    """
    Applies a "flip coin" mechanism to each item in the input domain.
    For each item, with probability 'probability', the original item is kept.
    Otherwise, it is replaced with a random integer between min and max of the domain.

    Parameters:
        probability (float): Probability (between 0 and 1) to keep the original item.
        domain: Input data (list, numpy array, or tensor).

    Returns:
        Data with each item either preserved or replaced with a random value,
        in the same format as the input.
    """
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")
    
    # Convert input to flat list and track original shape/type
    converter = TypeConverter(domain)
    items, _ = converter.get()

    # Precompute min and max for random replacement
    item_min = min(items)
    item_max = max(items)

    # Decide which items to keep vs replace
    keep_mask = [np.random.rand() < probability for _ in items]

    # Build result list by flipping coin
    result = [
        n if keep else random.randint(item_min, item_max)
        for n, keep in zip(items, keep_mask)
    ]

    # Restore result to original type/shape
    return converter.restore(result)
# -------------------------------
# Source: Differential Privacy by Cynthia Dwork, International Colloquium on Automata, Languages and Programming (ICALP) 2006, p. 1–12. doi:10.1007/11787006_1
# -------------------------------

def applyDPGaussian(domain, delta=1e-5, epsilon=0.1, gamma=1.0, accountant=None):
    """
    Applies Gaussian noise to the input data for differential privacy,
    and optionally tracks budget via a BudgetAccountant.

    Parameters:
        domain: Input data (list, numpy array, or tensor).
        delta (float): Failure probability (default: 1e-5).
        epsilon (float): Privacy parameter (default: 0.1).
        gamma (float): Scaling factor for noise (default: 1.0).
        accountant (BudgetAccountant, optional): Tracks spend for (ε, δ).

    Returns:
        Data with added Gaussian noise in the same format as the input.
    """
    # Convert to flat list and store type/shape
    converter = TypeConverter(domain)
    flat_list, _ = converter.get()

    # Compute σ for (ε, δ)-Gaussian mechanism
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * gamma / epsilon
    
    # Add Gaussian noise
    privatized = np.array(flat_list) + np.random.normal(loc=0, scale=sigma, size=len(flat_list))
    return converter.restore(privatized.tolist())


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
        Data with added Gaussian noise in the original format.
    """
    # Flatten input and track type/shape
    converter = TypeConverter(domain)
    data, _ = converter.get()

    # Calculate noise scale sigma
    sigma = np.sqrt((sensitivity ** 2 * alpha) / (2 * epsilon_bar))

    # Generate Gaussian noise
    noise = np.random.normal(loc=0, scale=sigma, size=len(data)) 

    # Add noise to data
    privatized = np.array(data) + noise

    # Restore to original type and shape
    return converter.restore(privatized.tolist())

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
    # Flatten input and track type/shape
    converter = TypeConverter(domain)
    data, _ = converter.get()

    scale = sensitivity * gamma / epsilon

    # Generate symmetric exponential noise: exponential with random sign
    noise = np.random.exponential(scale=scale, size=len(data))
    signs = np.random.choice([-1, 1], size=len(data))
    noise *= signs

    # Add noise to data
    privatized = np.array(data) + noise

    # Restore to original format
    return converter.restore(privatized.tolist())

# -------------------------------
# Source: Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis.
# In Proceedings of the Third Conference on Theory of Cryptography, TCC'06, 265–284. Berlin, Heidelberg, 2006. Springer-Verlag.
# URL: https://doi.org/10.1007/11681878_14, doi:10.1007/11681878_14.
# -------------------------------

def applyDPLaplace(domain, sensitivity=1, epsilon=0.01, gamma=1, accountant=None):
    """
    Applies Laplace noise to the input data for differential privacy.
    Tracks privacy budget with an optional BudgetAccountant.

    Parameters:
        domain: Input data (list, numpy array, or tensor).
        sensitivity: Maximum change by a single individual's data (default: 1).
        epsilon: Privacy parameter (default: 0.01).
        gamma: Scaling factor for noise (default: 1).
        accountant (BudgetAccountant, optional): Tracks spend for (ε, δ).

    Returns:
        Data with added Laplace noise in the same format as the input.
    """
    # Convert input to flat list and track original type/shape
    converter = TypeConverter(domain)
    data, _ = converter.get()

    # Calculate noise scale
    scale = sensitivity * gamma / epsilon

    # Add Laplace noise
    privatized = np.array(data) + np.random.laplace(loc=0, scale=scale, size=len(data))

    # Restore to original type/shape
    return converter.restore(privatized.tolist())
# -------------------------------
# Pruning Functions
# Source: https://arxiv.org/pdf/2311.06839.pdf
# Implementation: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700324.pdf
# -------------------------------
def applyPruning(domain, prune_ratio):
    """
    Applies pruning to reduce the magnitude of values.
    Values with absolute value below prune_ratio may be set to 0 or
    pruned to +/- prune_ratio based on a randomized threshold.

    Parameters:
        domain: Input data (list, numpy array, or tensor).
        prune_ratio (float): Threshold below which values are pruned.

    Returns:
        Pruned data in the same format as the input.
    """
    # Flatten and track type/shape
    converter = TypeConverter(domain)
    values, _ = converter.get()

    pruned = []
    for v in values:
        abs_v = abs(v)
        if abs_v < prune_ratio:
            rnd_tmp = random.random()
            # Randomized pruning logic
            if abs_v > rnd_tmp * prune_ratio:
                pruned.append(prune_ratio if v > 0 else -prune_ratio)
            else:
                pruned.append(0)
        else:
            pruned.append(v)  # Keep original if above prune_ratio

    # Restore pruned list to original input format
    return converter.restore(pruned)
# -------------------------------
# Source: https://arxiv.org/pdf/2311.06839.pdf
# -------------------------------

def applyPruningAdaptive(domain):
    """
    Applies adaptive pruning by determining a dynamic prune ratio.
    The prune ratio is set as the maximum absolute value plus a small constant.

    Parameters:
        domain: Input data (list, numpy array, or tensor).

    Returns:
        Adaptively pruned data in the original format.
    """
    # Flatten input and track type/shape
    converter = TypeConverter(domain)
    values, _ = converter.get()

    prune_ratio = max(abs(v) for v in values) + 0.1  # Adaptive prune threshold

    pruned = []
    for v in values:
        abs_v = abs(v)
        if abs_v < prune_ratio:
            rnd_tmp = random.random()
            if abs_v > rnd_tmp * prune_ratio:
                pruned.append(prune_ratio if v > 0 else -prune_ratio)
            else:
                pruned.append(0)
        else:
            pruned.append(v)  # Keep original if above prune_ratio

    # Restore pruned list to original input format
    return converter.restore(pruned)
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
        Differentially private pruned data in the original format.
    """
    # Flatten and track type/shape
    converter = TypeConverter(domain)
    values, _ = converter.get()

    # Apply pruning (assuming applyPruning accepts a flat list)
    pruned_values = applyPruning(values, prune_ratio)

    # Add Laplace noise
    noise_scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=noise_scale, size=len(pruned_values))
    privatized = np.array(pruned_values) + noise

    # Restore to original format
    return converter.restore(privatized.tolist())
#-----------Jackie work ---------

# -------------------------------
# Source: https://github.com/nikitaivkin/csh#
# -------------------------------
### Count Sketch for Private Aggregation

# Here is the new function, `applyCountSketch`, which uses the `CSVec` library to sketch and un-sketch your input data. This method is valuable for private aggregation in distributed settings like federated learning because it allows you to represent large, high-dimensional vectors (like model updates) with a much smaller sketch while still maintaining a strong estimate of the original data.

# def applyCountSketch(domain, sketch_rows, sketch_cols, dp_mechanism, dp_params, num_blocks=1, device=None):
#     """
#     Applies Count Sketch to the input domain, then adds differential privacy
#     noise to the sketched representation, and finally reconstructs the data.

#     Parameters:
#         domain: Input data (list, numpy array, or torch.Tensor).
#         sketch_rows (int): Number of rows (r) for the Count Sketch.
#         sketch_cols (int): Number of columns (c) for the Count Sketch.
#         dp_mechanism (callable): The differential privacy noise function to apply
#                                  (e.g., DP_Mechanisms.applyDPGaussian,
#                                  DP_Mechanisms.applyDPLaplace).
#                                  This function should expect a numpy array and return a numpy array.
#         dp_params (dict): A dictionary of parameters required by the dp_mechanism.
#         num_blocks (int, optional): Number of blocks for CSVec's memory optimization. Default is 1.
#         device (torch.device or str, optional): The device to perform operations on.
#                                                 Defaults to 'cuda' if available, else 'cpu'.

#     Returns:
#         The differentially private, sketched, and reconstructed data in the
#         original format of the input domain.
#     """
#     # Use TypeConverter to handle various input types and flatten the domain
#     converter = TypeConverter(domain)
#     flattened_data, original_shape = converter.get()

#     # Ensure flattened_data is a PyTorch tensor on the correct device for CSVec
#     if not isinstance(flattened_data, torch.Tensor):
#         flattened_data = torch.tensor(flattened_data, dtype=torch.float32)
    
#     if device is None:
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     flattened_data = flattened_data.to(device)

#     # Initialize CSVec
#     # d: original_dim, c: columns, r: rows
#     csvec_instance = CSVec(
#         d=flattened_data.numel(),
#         c=sketch_cols,
#         r=sketch_rows,
#         numBlocks=num_blocks,
#         device=device
#     )

#     # Accumulate the flattened data into the Count Sketch
#     csvec_instance.accumulateVec(flattened_data)

#     # Apply DP noise to the internal table of the CSVec
#     # The table is a (r, c) tensor. We need to detach, move to CPU, convert to numpy,
#     # apply noise, convert back to tensor, and move back to device.
#     sketched_table_np = csvec_instance.table.detach().cpu().numpy()
#     noisy_sketched_table_np = dp_mechanism(sketched_table_np, **dp_params)
#     csvec_instance.table = torch.tensor(noisy_sketched_table_np, dtype=torch.float32).to(device)

#     # Reconstruct the data from the noisy sketch
#     reconstructed_noisy_data = csvec_instance._findAllValues()

#     # Restore the reconstructed data to the original input type and shape
#     # _findAllValues returns a 1D tensor, so convert to list for TypeConverter
#     return converter.restore(reconstructed_noisy_data.tolist())

def applyCountSketch(
    domain: list | np.ndarray | torch.Tensor,
    num_rows: int,
    num_cols: int,
    epsilon: float,
    delta: float,
    mechanism: str = "gaussian",
    sensitivity: float = 1.0,
    gamma: float = 0.01,
    num_blocks: int = 1,
    device: torch.device | str | None = None
) -> list | np.ndarray | torch.Tensor:
    """
    Applies Count Sketch to the input data, then adds differential privacy
    noise to the sketched representation, and finally reconstructs the data.
    Consumes budget from the provided BudgetAccountant.
    """
    converter = TypeConverter(domain)
    flattened_data_tensor, original_shape = converter.get()

    # Ensure tensor
    if not isinstance(flattened_data_tensor, torch.Tensor):
        flattened_data_tensor = torch.tensor(flattened_data_tensor, dtype=torch.float32)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    flattened_data_tensor = flattened_data_tensor.to(device)

    csvec_instance = CSVec(
        d=flattened_data_tensor.numel(),
        c=num_cols,
        r=num_rows,
        numBlocks=num_blocks,
        device=device
    )

    csvec_instance.accumulateVec(flattened_data_tensor)

    sketched_table_np = csvec_instance.table.detach().cpu().numpy()

    if mechanism == "gaussian":
        noisy_sketched_table_np = applyDPGaussian(
            sketched_table_np, delta=delta, epsilon=epsilon, gamma=gamma
        )
    elif mechanism == "laplace":
        noisy_sketched_table_np = applyDPLaplace(
            sketched_table_np, sensitivity=sensitivity, epsilon=epsilon, gamma=gamma
        )
    else:
        raise ValueError(f"Unsupported DP mechanism for Count Sketch: {mechanism}. Choose 'gaussian' or 'laplace'.")

    csvec_instance.table = torch.tensor(noisy_sketched_table_np, dtype=torch.float32).to(device)
    reconstructed_noisy_data = csvec_instance._findAllValues()

    return converter.restore(reconstructed_noisy_data.tolist())