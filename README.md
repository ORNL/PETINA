

# PETINA: Privacy prEservaTIoN Algorithms

[![Python](https://img.shields.io/pypi/pyversions/petina)](https://pypi.org/project/PETINA/)
[![PyPI version](https://img.shields.io/pypi/v/petina)](https://pypi.org/project/PETINA/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Downloads](https://static.pepy.tech/badge/petina)](https://pepy.tech/project/petina)

**PETINA** is a general-purpose Python library for Differential Privacy (DP), designed for flexibility, modularity, and extensibility across a wide range of ML and data processing pipelines. It supports both numerical and categorical data, with tools for supervised and unsupervised tasks.

##  Features

PETINA includes state-of-the-art tools for:

###  Differential Privacy (DP) Mechanisms
- Laplace Mechanism
- Gaussian Mechanism
- Renyi-Gaussian Mechanism
- Exponential Mechanism
- Pruning
- Pruning Adaptive
- Pruning DP
- Count Sketch

###  Clipping
- Adaptive Clipping
- Clipping
- Clipping DP

### Encoding and Pertubation
- Pertubation
- Aggregation & Estimation
- Parameter Utilities
- Encoding

### Data Conversion Helper
- Flatten NumPy array to list and get shape
- Reshape list to NumPy array with shape
- Flatten PyTorch tensor to list and get shape
- Reshape list to PyTorch tensor with shape
- Detect input type and flatten to list with shape
- Convert list back to original data type and shape

## Quick Start

Below is a real world example when adding noise to age of various person
```python
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
```
We also provide hands-on [examples](./PETINA/examples/) in the examples folder.

- [Example 1](./PETINA/examples/1_basic.py): Basic PETINA Usage. This example script demonstrates key components of PETINA by generating synthetic data, configuring differential privacy parameters, and applying multiple DP mechanisms (Laplace, Gaussian, Exponential), encoding schemes (Unary, Histogram), clipping techniques (static and adaptive), pruning methods (fixed, adaptive, and DP-aware), and utility functions for calibrating privacy-preserving noise.
- [Example 2](./PETINA/examples/2_Personal_data.py): This script demonstrates how to apply PETINA’s differential privacy mechanisms—including unary encoding, Laplace noise, clipping, and pruning—to categorical and numerical features from the UCI Adult dataset for privacy-preserving data analysis.
- [Example 3](./PETINA/examples/3_Iris_data.py): This script applies PETINA's differential privacy techniques—including unary encoding for categorical species data, Laplace noise for numeric features, and adaptive clipping—to the Iris dataset for privacy-preserving data transformation and analysis.
- [Example 4](./PETINA/examples/4_ML_MNIST_No_MA.py): This script trains a CNN on MNIST with and without differential privacy using PETINA, supporting standard (Laplace, Gaussian) and Count Sketch-based mechanisms for privatizing gradients, and compares their impact on model performance and runtime.
- [Example 5](./PETINA/examples/5_ML_MNIST_MA_Opacus.py): This script trains a CNN on MNIST using PETINA with Gaussian differential privacy and budget accounting (via Opacus GDP accountant), optionally enhanced with Count Sketch compression (CSVec), and evaluates the privacy-utility tradeoff over multiple training runs.
- [Example 6](./PETINA/examples/6_FL_MNIST_MA_Opacus.py): This script implements federated training of a CNN on MNIST with optional Gaussian differential privacy and Count Sketch compression (CSVec), managing local client updates, DP noise addition, sketching, and secure aggregation with privacy budget accounting over multiple global rounds.
##  Installation
- Install from PyPI
```bash
pip install PETINA
```
- Install from Source
```bash
git clone https://github.com/ORNL/PETINA.git
cd PETINA
pip install -e .
```
## Citing PETINA
If you use PETINA in your research, please cite the official DOE OSTI release:  
> [https://www.osti.gov/doecode/biblio/149859](https://www.osti.gov/doecode/biblio/149859)
```bash
@misc{ doecode_149859,
  title = {ORNL/PETINA},
  author = {Kotevska, Ole and Nguyen, Duc},
  abstractNote = {This is a library that has implementation of privacy preservation algorithms.},
}
```
## Contributors
- Oliver Kotevska – KOTEVSKAO@ORNL.GOV – Maintainer
- Trong Nguyen – NT9@ORNL.GOV – Developer


We welcome community contributions to PETINA.

For major changes, please open an issue first. For small fixes or enhancements, submit a pull request. Include/update tests where applicable.

Contact: KOTEVSKAO@ORNL.GOV

## License
This project is licensed under the MIT License.

## Acknowledgements
This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research under Contract No. DE-AC05-00OR22725. This manuscript has been authored by UT-Battelle, LLC under Contract No. DE-AC05-00OR22725 with the U.S. Department of Energy. The United States Government retains and the publisher, by accepting the article for publication, acknowledges that the United States Government retains a non-exclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this manuscript, or allow others to do so, for United States Government purposes. The Department of Energy will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan (http://energy.gov/downloads/doe-public-access-plan).

