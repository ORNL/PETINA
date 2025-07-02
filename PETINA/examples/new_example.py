# File: real_world_demo_petina.py

from PETINA import DP_Mechanisms, Encoding_Pertubation, Clipping, Pruning
import numpy as np
import random

# --- Real-world data: Users' ages from a survey ---
user_ages = [23, 35, 45, 27, 31, 50, 29, 42, 38, 33]
print("Original ages:", user_ages)

# --- DP parameters ---
sensitivity = 1  # Age changes by 1 at most for neighboring datasets
epsilon = 0.5    # Moderate privacy budget
delta = 1e-5
gamma = 0.001

# --- Add Laplace noise to ages ---
noisy_ages = DP_Mechanisms.applyDPLaplace(user_ages, sensitivity, epsilon)
print("\nNoisy ages with Laplace Mechanism:")
print(noisy_ages)

# --- Encode noisy ages using Unary Encoding ---
p = Encoding_Pertubation.get_p(epsilon)
q = Encoding_Pertubation.get_q(p, epsilon)
encoded_ages = Encoding_Pertubation.unaryEncoding(noisy_ages, p=p, q=q)
print("\nUnary encoded noisy ages:")
print(encoded_ages)

# --- Summary ---
print("\nSummary:")
print(f"Original ages: {user_ages}")
print(f"Noisy ages: {np.round(noisy_ages, 2)}")
