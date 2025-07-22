import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from PETINA import DP_Mechanisms, Encoding_Pertubation, Clipping
from collections import Counter

# Load Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Convert species to list for unary encoding
species_list = data['species'].tolist()

# DP parameters
sensitivity = 1
epsilon = 1
delta = 1e-5
gamma = 1e-5

# --- Unary Encoding with DP on species ---
print("Original species (sample):", species_list[:10])

original_counts = Counter(species_list)
print("=== Original species counts ===")
for species, count in original_counts.items():
    print(f"{species}: {count}")
print("\nUnary encoding (species) with DP:")
privatized_species_counts = Encoding_Pertubation.unaryEncoding(species_list, p=0.75, q=0.25)
print(privatized_species_counts)

# --- Apply DP Laplace mechanism to each numerical feature ---
for feature in iris.feature_names:
    values = data[feature].tolist()
    print(f"\nOriginal '{feature}' (first 10 values):")
    print(values[:10])
    privatized_values = DP_Mechanisms.applyDPLaplace(values, sensitivity=sensitivity, epsilon=epsilon)
    print(f"DP Laplace mechanism on '{feature}' (first 10 values):")
    print(privatized_values[:10])

# --- Optional: Adaptive clipping on sepal length ---
print("\nOriginal sepal length (first 10 values):")
print(data['sepal length (cm)'].tolist()[:10])
clipped_sepal_length = Clipping.applyClippingAdaptive(data['sepal length (cm)'].tolist())
print("Adaptive clipping on sepal length (first 10 values):")
print(clipped_sepal_length[:10])

# #-----------OUTPUT------------
# Original species (sample): ['setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa']
# === Original species counts ===
# setosa: 50
# versicolor: 50
# virginica: 50

# Unary encoding (species) with DP:
# [('virginica', np.float64(49.0)), ('setosa', np.float64(57.0)), ('versicolor', np.float64(61.0))]     

# Original 'sepal length (cm)' (first 10 values):    
# [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9] 
# DP Laplace mechanism on 'sepal length (cm)' (first 10 values):
# [3.8741837104371033, 3.489575534734926, 5.795526513179814, 8.353078474880178, 4.83876789582926, 5.536928263610387, 4.329255195428825, 7.590582243024569, 5.745498151363301, 1.9637269245885034]

# Original 'sepal width (cm)' (first 10 values):     
# [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1] 
# DP Laplace mechanism on 'sepal width (cm)' (first 10 values):
# [4.185723138580048, 2.104333154208848, 2.5060389299258334, 3.008121095610164, 4.251165942144676, 3.179641894286568, 3.010560816533241, 1.8646169026429835, 5.324981170458997, 3.707596759384481]

# Original 'petal length (cm)' (first 10 values):    
# [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5] 
# DP Laplace mechanism on 'petal length (cm)' (first 10 values):
# [0.6205014216821986, -0.9895756422789161, 1.0291623123016227, 0.9087346440560822, -0.6832125041844503, 0.9139248060756305, -0.8341070319565387, 2.001269581131372, 0.8192617486607996, 1.0206715131540538]  

# Original 'petal width (cm)' (first 10 values):     
# [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1] 
# DP Laplace mechanism on 'petal width (cm)' (first 10 values):
# [1.8683611354164995, 0.6210078436514607, 0.18875909417105283, -0.4771252816183375, 1.8014388837807793, 0.34217044881573555, 2.1791468501340656, -0.11429181020335988, 1.6785993570024889, -0.5241579935794256]

# Original sepal length (first 10 values):
# [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9] 
# Adaptive clipping on sepal length (first 10 values):
# [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.6, 4.9]