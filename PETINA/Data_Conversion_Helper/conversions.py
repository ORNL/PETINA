# import math
# import random
# from scipy import stats as st
import torch
import numpy as np

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