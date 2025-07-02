# File: summer_2025/improving_petina/PETINA/PETINA/examples/tutorial1_basic.py

# --- Import necessary modules ---
from PETINA import DP_Mechanisms, Encoding_Pertubation, Clipping, Pruning
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

print("\n=== Sparse Vector Technique (Above Threshold SVT) ===")
print("DP =", DP_Mechanisms.above_threshold_SVT(0.3, domain, T=0.5, epsilon=epsilon))

print("\n=== Percentile Privacy ===")
print("Percentile Privacy =", DP_Mechanisms.percentilePrivacy(domain, 10))

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
print(Pruning.applyPruning(domain, 0.8))

print("Adaptive pruning:")
print(Pruning.applyPruningAdaptive(domain))

print("Pruning with DP (threshold=0.8):")
print(Pruning.applyPruningDP(domain, 0.8, sensitivity, epsilon))

# --- Utility Functions for Parameters ---
print("\n=== Utility Functions ===")
print("Get p from epsilon:")
print(Encoding_Pertubation.get_p(epsilon))

print("Get q from p and epsilon:")
print(Encoding_Pertubation.get_q(p=0.5, eps=epsilon))

print("Get gamma and sigma from p and epsilon:")
print(Encoding_Pertubation.get_gamma_sigma(p=0.5, eps=epsilon))
