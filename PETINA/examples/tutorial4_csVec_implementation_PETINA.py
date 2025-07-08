from PETINA import Sketching
import numpy as np
# -------------------------------
import math
import random
import torch
import numpy as np
from scipy import stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
import numpy as np
# from tqdm import tqdm
import matplotlib.pyplot as plt
from PETINA import CSVec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
# Example usage:
# First, create some sample data.
data_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
data_np = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
data_torch = torch.tensor([100, 200, 300], dtype=torch.float32)

# Define sketch dimensions.
rows = 10
cols = 1024

# Apply the Count Sketch to each data type.
sketched_list = Sketching.applyCountSketch(data_list, rows, cols)
sketched_np = Sketching.applyCountSketch(data_np, rows, cols)
sketched_torch = Sketching.applyCountSketch(data_torch, rows, cols)

# Print the results to see the approximation.
print("\n--- Results ---")
print("Original List:", data_list)
print("Sketched List:", sketched_list)
print("\nOriginal NumPy Array:", data_np)
print("Sketched NumPy Array:", sketched_np)
print("\nOriginal Torch Tensor:", data_torch)
print("Sketched Torch Tensor:", sketched_torch)