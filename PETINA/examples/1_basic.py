# --- Import necessary modules ---
from PETINA import DP_Mechanisms, Encoding_Pertubation, Clipping
import numpy as np
import random

random.seed(42)  # For reproducibility
# --- Generate synthetic data ---
base_domain = list(range(1, 11))  # Multiplier base
domain = [random.randint(10, 1000) * random.choice(base_domain) for _ in range(10)]
print("=== Synthetic Like Numbers ===")
print("Domain:", domain)

# --- Set DP parameters ---
sensitivity = 1
epsilon = 0.001
delta = 1e-5
gamma = 1e-5

# --- Differential Privacy Mechanisms ---
print("\n=== Laplace Mechanism ===")
print("DP =", DP_Mechanisms.applyDPLaplace(domain, sensitivity, epsilon))

print("\n=== Gaussian Mechanism ===")
print("DP =", DP_Mechanisms.applyDPGaussian(domain, delta, epsilon, gamma))

print("\n=== Exponential Mechanism ===")
print("DP =", DP_Mechanisms.applyDPExponential(domain, sensitivity, epsilon, gamma))


# --- Encoding Techniques ---
print("\n=== Unary Encoding ===")
print("Unary encoding (p=0.75, q=0.25):")
print(Encoding_Pertubation.unaryEncoding(domain, p=0.75, q=0.25))

print("\n=== Histogram Encoding ===")
print("Histogram encoding (version 1):")
print(Encoding_Pertubation.histogramEncoding(domain))

print("Histogram encoding (version 2):")
print(Encoding_Pertubation.histogramEncoding_t(domain))

# --- Clipping Techniques ---
print("\n=== Clipping ===")
print("Fixed clipping (min=0.4, max=1.0, step=0.1):")
print(Clipping.applyClippingDP(domain, 0.4, 1.0, 0.1))

print("Adaptive clipping:")
print(Clipping.applyClippingAdaptive(domain))

# --- Pruning Techniques ---
print("\n=== Pruning ===")
print("Fixed pruning (threshold=0.8):")
print(DP_Mechanisms.applyPruning(domain, 0.8))

print("Adaptive pruning:")
print(DP_Mechanisms.applyPruningAdaptive(domain))

print("Pruning with DP (threshold=0.8):")
print(DP_Mechanisms.applyPruningDP(domain, 0.8, sensitivity, epsilon))

# --- Utility Functions for Parameters ---
print("\n=== Utility Functions ===")
print("Get p from epsilon:")
print(Encoding_Pertubation.get_p(epsilon))

print("Get q from p and epsilon:")
print(Encoding_Pertubation.get_q(p=0.5, eps=epsilon))

print("Get gamma and sigma from p and epsilon:")
print(Encoding_Pertubation.get_gamma_sigma(p=0.5, eps=epsilon))


# #-----------OUTPUT------------

# === Synthetic Like Numbers ===
# Domain: [1328, 175, 1040, 304, 6318, 990, 442, 80, 932, 5270]

# === Laplace Mechanism ===
# DP = [1320.62177961  187.46655673 1033.11533695  275.22139055 6333.68691776
#   994.16487497  457.88459267   84.90268961  946.33431801 5260.64490999]

# === Gaussian Mechanism ===
# DP = [1328.0053238228013, 175.0004720699406, 1039.9088404779786, 303.98554147404155, 6317.983805639122, 990.0524320700276, 441.9779219277438, 79.99875144738958, 931.9742725741538, 5269.952955179929]

# === Exponential Mechanism ===
# DP = [1327.9936220914026, 174.97332006392915, 1040.0119082825654, 304.0283981744525, 6317.99439136121, 989.9744648449474, 442.0230133161982, 80.0191800651194, 931.9780781645926, 5270.030814489527]

# === Unary Encoding ===
# Unary encoding (p=0.75, q=0.25):
# [(932, 3.0), (6318, 1.0), (175, 5.0), (1328, 7.0), (1040, -1.0), (304, -1.0), (80, -1.0), (5270, 1.0), (442, -1.0), (990, -3.0)]

# === Histogram Encoding ===
# Histogram encoding (version 1):
# [-377.7142305747353, -17.551272500549132, 71.67769402438955, 6.7975412961143995, -216.77400357756073, 307.8244337471643, 79.4785660392272, -223.4042841405975, 76.13356950219686, 56.817256548948095]
# Histogram encoding (version 2):
# [9, 174, -72, -72, 9, -31, -31, -31, -113, -113]

# === Clipping ===
# Fixed clipping (min=0.4, max=1.0, step=0.1):
# [1.5833066703267376, -8.998044088985806, -8.627863251838965, 4.420893038900671, 14.927067934956149, -10.839364535590612, -4.600061765151264, -69.50616111204471, 6.085625139530498, 8.230861468782194]
# Adaptive clipping:
# [1328.0, 175.0, 1040.0, 304.0, 6318.0, 990.0, 442.0, 122.75, 932.0, 5270.0]

# === Pruning ===
# Fixed pruning (threshold=0.8):
# [1328, 175, 1040, 304, 6318, 990, 442, 80, 932, 5270]
# Adaptive pruning:
# [6318.1, 0, 0, 0, 6318.1, 0, 0, 6318.1, 0, 6318.1]
# Pruning with DP (threshold=0.8):
# [1334.617713354503, 164.6872581551384, 1028.7538449049823, 281.06717278991414, 6314.503225296515, 981.5160683596098, 442.350410242875, 82.41855113978585, 916.4173039668901, 5274.777392718558]

# === Utility Functions ===
# Get p from epsilon:
# 0.51
# Get q from p and epsilon:
# 0.47502081252106
# Get gamma and sigma from p and epsilon:
# (0.06265450610580386, 25.073720567096746)