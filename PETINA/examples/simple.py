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
#------OUTPUT------
# Original ages: [23, 35, 45, 27, 31, 50, 29, 42, 38, 33]

# Noisy ages with Laplace Mechanism:
# [21.46703958 34.93585449 47.36478841 25.68077936 30.11460444 49.3448666
#  28.8128474  36.54981691 37.6103979  33.32033856]

# Unary encoded noisy ages:
# [(33.320338556461415, np.float64(14.023220368761203)), (34.935854491045006, np.float64(5.97677963123879)), (36.54981690878978, np.float64(22.06966110628362)), (37.61039790139999, np.float64(-10.116101843806039)), (47.36478841495265, np.float64(-18.162542581328452)), (49.34486659855414, np.float64(14.023220368761203)), (21.467039579955127, np.float64(-18.162542581328452)), (25.6807793619914, np.float64(-2.069661106283625)), (28.812847396103876, np.float64(5.97677963123879)), (30.114604444236978, np.float64(-10.116101843806039))]

# Summary:
# Original ages: [23, 35, 45, 27, 31, 50, 29, 42, 38, 33]
# Noisy ages: [21.47 34.94 47.36 25.68 30.11 49.34 28.81 36.55 37.61 33.32]