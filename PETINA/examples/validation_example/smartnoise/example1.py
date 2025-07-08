# # -----------------------------------
# # SmartNoise Laplace Example
# # -----------------------------------
# from snsql.sql._mechanisms import Laplace
# import numpy as np

# # Input data and privacy parameters
# data = [10.0, 20.0, 30.0]
# epsilon = 1.0

# # Define Laplace mechanism with bounds for sensitivity estimation
# lap = Laplace(epsilon=epsilon, lower=0.0, upper=30.0)

# # Release noisy results
# noisy_smartnoise = lap.release(data)
# print("SmartNoise:", noisy_smartnoise)

# # -----------------------------------
# # PETINA Laplace Example (matching SmartNoise)
# # -----------------------------------
# from PETINA import DP_Mechanisms

# # Use the same data and epsilon
# data = [10.0, 20.0, 30.0]
# epsilon = 1.0

# # Reuse the noise scale computed by SmartNoise
# sensitivity = lap.scale  # this is the internally computed scale

# # Apply Laplace mechanism using PETINA with the same scale
# noisy_petina = DP_Mechanisms.applyDPLaplace(data, sensitivity=sensitivity, epsilon=epsilon)
# print("PETINA:", noisy_petina)
# import opendp.prelude as dp
# dp.enable_features("contrib")
# input_space = dp.atom_domain(T=float), dp.absolute_distance(T=float)
# laplace = dp.m.make_laplace(*input_space, scale=1.0)
# print('100?', laplace(100.0))

# from PETINA import DP_Mechanisms

# # True value to privatize
# true_value = 100.0
# data = [true_value]

# # Apply PETINA's Laplace mechanism
# noisy_data = DP_Mechanisms.applyDPLaplace(
#     data,
#     epsilon=1.0,
#     sensitivity=1.0
# )

# print("Noisy value:", noisy_data[0])
import numpy as np
# np.random.seed(45)  # Fix NumPy RNG for any numpy-based randomness (PETINA may use its own RNG)

# OpenDP imports
import opendp.prelude as dp
dp.enable_features("contrib")

# PETINA import
from PETINA import DP_Mechanisms

# Define input domain and metric for OpenDP Laplace
input_space = dp.vector_domain(dp.atom_domain(T=float)), dp.l1_distance(T=float)

# Create OpenDP Laplace mechanism with scale = 1.0
laplace = dp.m.make_laplace(*input_space, scale=1.0)

# Input values to privatize
data = [10.0, 20.0, 30.0]

# Number of trials to run for statistics
num_trials = 100000

# Containers to hold noisy outputs
noisy_opendp_results = []
noisy_petina_results = []

for _ in range(num_trials):
    # OpenDP noisy release
    noisy_opendp = laplace(data)
    noisy_opendp_results.append(noisy_opendp)

    # PETINA noisy release
    noisy_petina = DP_Mechanisms.applyDPLaplace(
        data,
        epsilon=1.0,
        sensitivity=1.0
    )
    noisy_petina_results.append(noisy_petina)

# Convert to numpy arrays for easier computation
noisy_opendp_results = np.array(noisy_opendp_results)
noisy_petina_results = np.array(noisy_petina_results)

# Compute means and std devs
opendp_mean = noisy_opendp_results.mean(axis=0)
opendp_std = noisy_opendp_results.std(axis=0)

petina_mean = noisy_petina_results.mean(axis=0)
petina_std = noisy_petina_results.std(axis=0)

print("OpenDP (SmartNoise) Laplace Noise Statistics:")
for i, (m, s) in enumerate(zip(opendp_mean, opendp_std)):
    print(f"  Coordinate {i}: mean = {m:.4f}, std = {s:.4f}")

print("\nPETINA Laplace Noise Statistics:")
for i, (m, s) in enumerate(zip(petina_mean, petina_std)):
    print(f"  Coordinate {i}: mean = {m:.4f}, std = {s:.4f}")
