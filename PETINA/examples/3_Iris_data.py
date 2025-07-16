import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from PETINA import DP_Mechanisms, Encoding_Pertubation, Clipping

# Load Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Convert species to list for unary encoding
species_list = data['species'].tolist()

# DP parameters
sensitivity = 1
epsilon = 0.5
delta = 1e-5
gamma = 0.01

# --- Unary Encoding with DP on species ---
print("Unary encoding (species) with DP:")
privatized_species_counts = Encoding_Pertubation.unaryEncoding(species_list, p=0.75, q=0.25)
print(privatized_species_counts)

# --- Apply DP Laplace mechanism to each numerical feature ---
for feature in iris.feature_names:
    values = data[feature].tolist()
    privatized_values = DP_Mechanisms.applyDPLaplace(values, sensitivity, epsilon)
    print(f"\nDP Laplace mechanism on '{feature}':")
    print(privatized_values[:10])  # show first 10 privatized values

# --- Optional: Adaptive clipping on sepal length ---
clipped_sepal_length = Clipping.applyClippingAdaptive(data['sepal length (cm)'].tolist())
print("\nAdaptive clipping on sepal length (first 10 values):")
print(clipped_sepal_length[:10])


#OUTPUT:
# Unary encoding (species) with DP:
# [('versicolor', 39.0), ('setosa', 63.0), ('virginica', 55.0)]

# DP Laplace mechanism on 'sepal length (cm)':
# [ 2.61708022  3.51163326 -1.82017246  3.46034212  7.97935142  4.54933603
#   4.41295874  4.30167443  4.80941638  5.83295261]

# DP Laplace mechanism on 'sepal width (cm)':
# [4.02578723 1.61426733 4.09015688 4.07077359 7.04745213 4.68124748
#  4.35683098 1.77620328 2.77462434 2.99384368]

# DP Laplace mechanism on 'petal length (cm)':
# [-1.07934579  1.26915955  1.52519904  4.1523995   1.18763398  3.32359533
#   1.09661959  1.07512602  1.64709652 -1.74361331]

# DP Laplace mechanism on 'petal width (cm)':
# [-1.23537807 -0.67660207  0.49552881 -0.03825378  3.31258487  0.63879834
#   3.73332148 -1.76443021  0.81447663  1.18774006]

# Adaptive clipping on sepal length (first 10 values):
# [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.6, 4.9]