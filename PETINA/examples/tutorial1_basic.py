# --- Import necessary modules ---
from PETINA import DP_Mechanisms, Encoding_Pertubation, Clipping
import numpy as np
import random

# --- Generate synthetic data ---
base_domain = list(range(1, 11))  # Multiplier base
domain = [random.randint(10, 1000) * random.choice(base_domain) for _ in range(10)]
print("=== Synthetic Like Numbers ===")
print("Domain:", domain)

# --- Set DP parameters ---
sensitivity = 1
epsilon = 0.1
delta = 1e-4
gamma = 0.001

# --- Differential Privacy Mechanisms ---
print("\n=== Flip Coin Mechanism ===")
print("FlipCoin (p=0.9) on domain [1-10]:")
print(DP_Mechanisms.applyFlipCoin(probability=0.9, domain=[1,2,3,4,5,6,7,8,9,10]))

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


#-----------OUTPUT------------
# === Synthetic Like Numbers ===
# Domain: [4959, 178, 3096, 1280, 2850, 3514, 5856, 2108, 476, 2343]

# === Flip Coin Mechanism ===
# FlipCoin (p=0.9) on domain [1-10]:
# [8, 2, 3, 4, 5, 10, 7, 8, 3, 10]

# === Laplace Mechanism ===
# DP = [4966.77119037  175.73772536 3074.43586924 1284.75490937 2839.74647937
#  3506.55554218 5868.713818   2107.53989743  478.12108054 2323.685295  ]

# === Gaussian Mechanism ===
# DP = [4958.99027034  177.99684628 3096.0377101  1280.08355631 2850.02198864
#  3514.01370751 5856.01891116 2107.97750544  475.99153929 2342.97229384]

# === Exponential Mechanism ===
# DP = [4958.971160649393, 177.97795979740886, 3096.0012499871846, 1280.0469138218696, 2850.0194923168992, 3514.0025226040307, 5855.9825798208485, 2108.015480527959, 475.99692120123683, 2342.998286905031]

# === Unary Encoding ===
# Unary encoding (p=0.75, q=0.25):
# [(1280, 3.0), (5856, 3.0), (2850, -1.0), (2343, 3.0), (178, 5.0), (3096, -1.0), (3514, -1.0), (476, -1.0), (2108, 5.0), (4959, -1.0)]

# === Histogram Encoding ===
# Histogram encoding (version 1):
# [-40.06122873741014, -142.50061678508257, -27.6257765845182, 235.393289784596, -104.49254733063353, 293.90006514112497, 65.03806376533237, 20.403750293400947, -52.66399323401487, 186.2696332493757]
# Histogram encoding (version 2):
# [-31, 133, 92, -72, -31, -72, 92, 51, -72, 92]

# === Clipping ===
# Fixed clipping (min=0.4, max=1.0, step=0.1):
# [-7.2807478650703015, 3.136154338504969, 2.6132351359567516, -13.664195536678365, 2.481204409258468, 28.14970441032584, 0.21945348969260625, 0.14633249627639988, -7.014333674469619, -0.9799091697728971]
# Adaptive clipping:
# [4959.0, 312.1, 3096.0, 1280.0, 2850.0, 3514.0, 5856.0, 2108.0, 476.0, 2343.0]

# === Pruning ===
# Fixed pruning (threshold=0.8):
# []
# Adaptive pruning:
# [5856.1, 5856.1, 0, 0, 0, 5856.1, 5856.1, 0, 0, 0]
# Pruning with DP (threshold=0.8):
# []

# === Utility Functions ===
# Get p from epsilon:
# 0.51
# Get q from p and epsilon:
# 0.47502081252106
# Get gamma and sigma from p and epsilon:
# (0.06265450610580386, 25.073720567096746)