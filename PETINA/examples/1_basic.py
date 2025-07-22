# --- Import necessary modules ---
from PETINA import DP_Mechanisms, Encoding_Pertubation, Clipping
import numpy as np
import random
seed=42
random.seed(seed)  # For reproducibility
# --- Generate synthetic data ---
base_domain = list(range(1, 11))  # Multiplier base
domain = [random.randint(10, 1000) * random.choice(base_domain) for _ in range(10)]
print("=== Synthetic Like Numbers ===")
print("Domain:", domain)

# --- Set DP parameters ---
sensitivity = 1
epsilon = 1
delta = 1e-5
gamma = 1e-5
print(f"sensitivity: {sensitivity}")
print(f"epsilon: {epsilon}")
print(f"delta: {delta}")
print(f"gamma: {gamma}")
print(f"Random seed: {seed}")
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
print(Clipping.applyClippingDP(domain, 1.0, 0.1))

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
# sensitivity: 1
# epsilon: 1
# delta: 1e-05
# gamma: 1e-05
# Random seed: 42

# === Laplace Mechanism ===
# DP = [1329.1528397600607, 177.5495658537669, 1039.670544805219, 301.62507468588217, 6316.914663879961, 989.8619358878439, 441.6585098450425, 79.20231088742678, 929.9268121324433, 5269.959239576952]

# === Gaussian Mechanism ===
# DP = [1327.9999827460542, 175.0000011141145, 1039.9999860226922, 304.00000042350445, 6318.000025183321, 990.000018293934, 441.99997707059606, 80.00002715862725, 932.0000075881421, 5269.999971876684]

# === Exponential Mechanism ===
# DP = [1328.0000216389922, 174.99998419518755, 1040.0000091826528, 304.00000599670824, 6317.999995523504, 990.0000135199173, 441.9999919076529, 79.99999082023285, 932.0000040910651, 5270.000042680841]

# === Unary Encoding ===
# Unary encoding (p=0.75, q=0.25):
# [(932, np.float64(9.0)), (6318, np.float64(-1.0)), (175, np.float64(1.0)), (1328, np.float64(-3.0)), (1040, np.float64(1.0)), (304, np.float64(3.0)), (80, np.float64(-1.0)), (5270, np.float64(5.0)), (442, np.float64(1.0)), (990, np.float64(5.0))]

# === Histogram Encoding ===
# Histogram encoding (version 1):
# [np.float64(207.42203848142336), np.float64(44.52269976181345), np.float64(93.73384245110324), np.float64(340.187743714482), np.float64(171.529622546254), np.float64(214.21598362722432), np.float64(64.28437619306867), np.float64(-269.6466944092254), np.float64(182.09346428538552), np.float64(137.1017187531161)]
# Histogram encoding (version 2):
# [-154, 92, 92, 9, -31, 92, 51, -113, 92, -72]

# === Clipping ===
# [1323.096424099652, 171.9037907352158, 1053.026204369521, 283.57258401109675, 5851.7691079727, 987.4848350171588, 433.64609362658405, 143.6392114493155, 943.9047741873377, 5284.517284429717]
# Adaptive clipping:
# [1328.0, 175.0, 1040.0, 304.0, 6318.0, 990.0, 442.0, 122.75, 932.0, 5270.0]

# === Pruning ===
# Fixed pruning (threshold=0.8):
# [1328, 175, 1040, 304, 6318, 990, 442, 80, 932, 5270]
# Adaptive pruning:
# [6318.1, 0, 0, 0, 6318.1, 0, 0, 6318.1, 0, 6318.1]
# Pruning with DP (threshold=0.8):
# [1328.7405037218725, 174.65803419444615, 1038.2563360021106, 304.55775440929824, 6314.976024023202, 990.3039201120661, 445.85470224532105, 79.9608847667527, 931.6438631226038, 5270.361942908907]

# === Utility Functions ===
# Get p from epsilon:
# 0.59
# Get q from p and epsilon:
# 0.2689414213699951
# Get gamma and sigma from p and epsilon:
# (np.float64(0.6160176926724505), np.float64(2.5785779646204614))