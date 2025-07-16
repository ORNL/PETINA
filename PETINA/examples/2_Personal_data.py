# The example loads a real-world dataset and applies PETINA’s differential privacy techniques:
# For categorical data (education): It uses unary encoding with randomized response to add noise while preserving privacy, estimating counts of each category.
# For numerical data (age): It adds Laplace noise, clips large values, and prunes small values to protect individual data while keeping useful statistics.
import pandas as pd
import numpy as np
from PETINA import DP_Mechanisms, Encoding_Pertubation, Clipping

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
epsilon = 0.5
sensitivity = 1

# --- Apply Unary Encoding DP ---
print("\n=== Unary Encoding with DP ===")
privatized_counts = Encoding_Pertubation.unaryEncoding(education, p=0.75, q=0.25)
for edu_level, count in privatized_counts:
    print(f"{edu_level}: {count:.2f}")

# --- Apply Laplace DP on numerical feature 'age' ---
print("\n=== Laplace Mechanism on age ===")
ages = data['age'].tolist()
laplace_ages = DP_Mechanisms.applyDPLaplace(ages, sensitivity, epsilon)
print("Original ages (first 10):", ages[:10])
print("Privatized ages (first 10):", laplace_ages[:10])

# --- Clipping example ---
print("\n=== Clipping ages with threshold=60 ===")
clipped_ages = Clipping.applyClipping(ages, clipping_threshold=60)
print("Clipped ages (first 10):", clipped_ages[:10])

# --- Pruning example ---
print("\n=== Pruning ages with threshold=25 ===")
pruned_ages = DP_Mechanisms.applyPruning(ages, prune_ratio=25)
print("Pruned ages (first 10):", pruned_ages[:10])


#OUTPUT:
# Original education categories (sample): ['Bachelors', 'Bachelors', 'HS-grad', '11th', 'Bachelors', 'Masters', '9th', 'HS-grad', 'Masters', 'Bachelors']

# === Unary Encoding with DP ===
# Some-college: 7391.50
# Assoc-voc: 1465.50
# Doctorate: 227.50
# 7th-8th: 625.50
# 9th: 515.50
# Prof-school: 679.50
# 10th: 1129.50
# Assoc-acdm: 1239.50
# 12th: 771.50
# Masters: 1677.50
# Bachelors: 5445.50
# 11th: 1159.50
# 1st-4th: 131.50
# Preschool: 515.50
# HS-grad: 10427.50
# 5th-6th: 427.50

# === Laplace Mechanism on age ===
# Original ages (first 10): [39, 50, 38, 53, 28, 37, 49, 52, 31, 42]
# Privatized ages (first 10): [36.92346087 49.81399273 37.27722657 53.84530975 31.40371438 37.13417441
#  48.20773513 52.43283555 26.40484366 39.78241878]

# === Clipping ages with threshold=60 ===
# Clipped ages (first 10): [39, 50, 38, 53, 28, 37, 49, 52, 31, 42]

# === Pruning ages with threshold=25 ===
# Pruned ages (first 10): [39, 50, 38, 53, 28, 37, 49, 52, 31, 42]