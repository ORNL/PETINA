# The example loads a real-world dataset and applies PETINA’s differential privacy techniques:
# For categorical data (education): It uses unary encoding with randomized response to add noise while preserving privacy, estimating counts of each category.
# For numerical data (age): It adds Laplace noise, clips large values, and prunes small values to protect individual data while keeping useful statistics.
import pandas as pd
import numpy as np
from PETINA import DP_Mechanisms, Encoding_Pertubation, Clipping
from collections import Counter

# Download Adult dataset from UCI repository URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
col_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]
data = pd.read_csv(url, names=col_names, na_values=" ?", skipinitialspace=True)

# Drop missing values for simplicity
data = data.dropna()

# Select categorical feature: education
education = data['education'].tolist()
print(f"Original education categories (sample): {education[:10]}")

# DP parameters
epsilon = 1
sensitivity = 1

# Original counts
original_counts = Counter(education)
print("=== Original education counts ===")
for edu_level, count in original_counts.items():
    print(f"{edu_level}: {count}")

# DP Unary Encoding counts
print("\n=== Unary Encoding with DP ===")
privatized_counts = Encoding_Pertubation.unaryEncoding(education, p=0.75, q=0.25)
for edu_level, count in privatized_counts:
    print(f"{edu_level}: {count:.2f}")

# --- Apply Laplace DP on numerical feature 'age' ---
ages = data['age'].tolist()
print("\n=== Original ages (first 10): ===")
print(ages[:10])
laplace_ages = DP_Mechanisms.applyDPLaplace(ages, sensitivity, epsilon)
print("=== Privatized ages with Laplace (first 10): ===")
print(laplace_ages[:10])

# --- Clipping example with DP ---
print("\n=== Original ages for clipping (first 10): ===")
print(ages[:10])
clipped_ages = Clipping.applyClippingDP(ages, sensitivity=sensitivity, epsilon=epsilon)
print("=== Clipped ages with DP (first 10): ===")
print(clipped_ages[:10])

# --- Pruning example with DP ---
print("\n=== Original ages for pruning (first 10): ===")
print(ages[:10])
pruned_ages = DP_Mechanisms.applyPruningDP(ages, prune_ratio=0.8, sensitivity=sensitivity, epsilon=epsilon)
print("=== Pruned ages with DP (first 10): ===")
print(pruned_ages[:10])


# #-----------OUTPUT------------
# Original education categories (sample): ['Bachelors', 'Bachelors', 'HS-grad', '11th', 'Bachelors', 'Masters', '9th', 'HS-grad', 'Masters', 'Bachelors']
# === Original education counts ===
# Bachelors: 5355
# HS-grad: 10501
# 11th: 1175
# Masters: 1723
# 9th: 514
# Some-college: 7291
# Assoc-acdm: 1067
# Assoc-voc: 1382
# 7th-8th: 646
# Doctorate: 413
# Prof-school: 576
# 5th-6th: 333
# 10th: 933
# 1st-4th: 168
# Preschool: 51
# 12th: 433

# === Unary Encoding with DP ===
# 12th: 497.50
# 7th-8th: 743.50
# 5th-6th: 509.50
# Prof-school: 579.50
# Assoc-acdm: 1157.50
# 10th: 1097.50
# Bachelors: 5165.50
# Doctorate: 421.50
# HS-grad: 10623.50
# 11th: 917.50
# Preschool: 263.50
# Some-college: 7321.50
# Assoc-voc: 1221.50
# Masters: 1601.50
# 9th: 535.50
# 1st-4th: 569.50

# === Original ages (first 10): ===
# [39, 50, 38, 53, 28, 37, 49, 52, 31, 42]
# === Privatized ages with Laplace (first 10): ===
# [38.997128498279494, 51.57444538834403, 41.06291024083538, 52.93510581173242, 27.542016767958753, 40.081788088245126, 49.53149848581675, 52.10112386336452, 30.26981144932168, 43.33862507521245]

# === Original ages for clipping (first 10): ===
# [39, 50, 38, 53, 28, 37, 49, 52, 31, 42]
# === Clipped ages with DP (first 10): ===
# [36.14885900842233, 50.294193778875986, 37.486896801551325, 52.72039583789096, 27.804455352085384, 32.99794194681864, 49.11906923828846, 52.48797181436213, 31.53790534727279, 41.63536357880384]

# === Original ages for pruning (first 10): ===
# [39, 50, 38, 53, 28, 37, 49, 52, 31, 42]
# === Pruned ages with DP (first 10): ===
# [40.30377868415831, 49.28907245870514, 36.08212621267062, 55.59835372002478, 30.66513834199118, 37.044803145815976, 49.4231244587092, 52.624338301178184, 32.10680220777121, 42.0731093599918]