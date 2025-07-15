import torch
import numpy as np

# -------------------------------
# Data Conversion Helper Functions
# -------------------------------

def numpy_to_list(nd_array):
    """
    Flatten a NumPy array to a list and return its original shape.
    """
    return nd_array.flatten().tolist(), nd_array.shape


def list_to_numpy(flat_list, shape):
    """
    Reshape a flattened list back to a NumPy array of given shape.
    """
    return np.array(flat_list).reshape(shape)


def torch_to_list(tensor):
    """
    Flatten a PyTorch tensor to a list and return its original shape.
    """
    return tensor.flatten().tolist(), tensor.shape


def list_to_torch(flat_list, shape):
    """
    Reshape a flattened list back to a PyTorch tensor of given shape.
    """
    return torch.as_tensor(flat_list).reshape(shape)


def type_checking_and_return_lists(domain):
    """
    Convert input (tensor, ndarray, or list) to flattened list and return shape info.
    Shape is 0 for plain lists.
    """
    if isinstance(domain, torch.Tensor):
        return torch_to_list(domain)
    elif isinstance(domain, np.ndarray):
        return numpy_to_list(domain)
    elif isinstance(domain, list):
        return domain, 0
    else:
        raise TypeError("Input must be list, numpy.ndarray, or torch.Tensor")


def type_checking_return_actual_dtype(domain, result, shape):
    """
    Convert flattened processed list back to the original data type and shape.
    """
    if isinstance(domain, torch.Tensor):
        return list_to_torch(result, shape)
    elif isinstance(domain, np.ndarray):
        return list_to_numpy(result, shape)
    else:
        return result
