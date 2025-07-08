

# PETINA: Privacy prEservaTIoN Algorithms

[![Python](https://img.shields.io/pypi/pyversions/petina)](https://pypi.org/project/PETINA/)
[![PyPI version](https://img.shields.io/pypi/v/petina)](https://pypi.org/project/PETINA/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Downloads](https://static.pepy.tech/badge/petina)](https://pepy.tech/project/petina)

**PETINA** is a general-purpose Python library for Differential Privacy (DP), designed for flexibility, modularity, and extensibility across a wide range of ML and data processing pipelines. It supports both numerical and categorical data, with tools for supervised and unsupervised tasks.

##  Features

PETINA includes state-of-the-art tools for:

###  Differential Privacy Mechanisms
- Laplace, Gaussian, and Exponential Mechanisms
- Sparse Vector Technique (SVT)
- Percentile Mechanism
- Unary and Histogram Encoding

###  Sketching Algorithms
- Count Sketch
- Fast Projection-Based Sketching

###  Adaptive Privacy
- Adaptive Clipping
- Adaptive Pruning

###  Utility Functions
- Convert between Python list, NumPy array, and PyTorch tensor
- Type casting and validation
- Compute privacy parameters (e.g., `p`, `q`, `gamma`, `sigma`) from ε

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

- [Example 1](./PETINA/examples/tutorial1_basic.py): Basic PETINA Usage.
- [Example 2](./PETINA/examples/tutorial2_CountSketch_PureLDP.py): This example demonstrates how to perform frequency estimation on synthetic categorical data using various pure Local Differential Privacy (LDP) algorithms. It also compares PETINA's Count Mean Sketch (CMS) and CSVec-based sketching with baseline LDP methods and a centralized CMS variant.
- [Example 3](./PETINA/examples/tutorial3_Moment_Accounting.py): This script demonstrates differentially private training in PyTorch using PETINA with budget tracking, clipping, and noise injection.
- [Example 4](./PETINA/examples/tutorial4_csVec_implementation_PETINA.py): This script demonstrates how to apply PETINA's Count Sketch mechanism to lists, NumPy arrays, and PyTorch tensors for efficient data approximation.
- [Example 5](./PETINA/examples/tutorial5_PETINA_MA_Implement.py): This script runs a federated learning simulation with optional Laplace, Gaussian, or Count Sketch-based privacy mechanisms, integrated with PETINA's budget accountant to track and gracefully handle differential privacy budget consumption.
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

