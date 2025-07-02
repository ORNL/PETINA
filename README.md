<!-- 
# PETINA: Privacy prEservaTIoN Algorithms

**PETINA** is a Python package that provides a comprehensive suite of state-of-the-art differential privacy (DP) algorithms. It supports both supervised and unsupervised learning tasks and handles both numerical and categorical data types. PETINA is designed to be modular and easily integrated into existing machine learning pipelines, addressing common limitations in other libraries, such as limited algorithm diversity or complexity of use.

If you use this package in your work, please cite the appropriate reference from the official OSTI page:  
[https://www.osti.gov/doecode/biblio/149859](https://www.osti.gov/doecode/biblio/149859)

## Contributors

The PETINA project is developed and maintained by the following contributors:

- **Oliver Kotevska** – KOTEVSKAO@ORNL.GOV – Project Lead  
- **Trong Nguyen** – NT9@ORNL.GOV – Intern Student  

We welcome new contributors. See the [Contributing](#contributing) section for how to get involved.


## Featuresw

PETINA provides the following functionality:

### Differential Privacy Mechanisms
- Gaussian Mechanism (standard DP)
- Rényi Differential Privacy (RDP) Gaussian Mechanism
- Exponential Mechanism
- Laplace Mechanism
- Unary Encoding, Histogram Encoding
- Sparse Vector Technique
- DP Clipping, DP Pruning, DP Percentile Mechanism

### Sketching Algorithms
- Count Sketch
- Fast Projection-Based Sketching

### Adaptive Mechanisms
- Adaptive Clipping
- Adaptive Pruning

### Utility Functions
- Convert between:
  - Python list ↔ NumPy array
  - Python list ↔ PyTorch tensor
- Type casting and validation utilities
- Compute privacy parameters (p, q) from a given epsilon (ε)

## Installation

### Install via PyPI

```bash
pip install PETINA
````

### Install from Source

```bash
git clone https://github.com/ORNL/PETINA.git
cd PETINA
pip install -e .
```


## Quick Start

Here is a minimal example to apply the Gaussian mechanism:

```python
import numpy as np
from PETINA import DP_Mechanisms

domain = [1, 2, 13, 4, 5, 11, 21, 3, 14, 5, 10, 12, 4, 16, 7, 18, 10, 30, 20, 15, 27]
epsilon = 0.1
delta = 1e-5

print("DP Output =", DP_Mechanisms.applyDPGaussian(domain, delta, epsilon))
```


## More Examples

We provide additional [examples](./PETINA/examples/) in this folder to help you better understand Differential Privacy concepts and how to use the PETINA library effectively.

## License

This project is licensed under the [MIT License](LICENSE).


## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research under Contract No. DE-AC05-00OR22725. This manuscript has been co-authored by UT-Battelle, LLC under Contract No. DE-AC05-00OR22725 with the U.S. Department of Energy. The United States Government retains and the publisher, by accepting the article for publication, acknowledges that the United States Government retains a non-exclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this manuscript, or allow others to do so, for United States Government purposes. The Department of Energy will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan (http://energy.gov/downloads/doe-public-access-plan).


## Contributing

Contributions to PETINA are welcome.

If you're planning a major change, please open an issue first to discuss it. For smaller changes or fixes, feel free to open a pull request directly.

Make sure to update or add tests if needed.

For questions or to coordinate contributions, you may contact [Dr.Olivera Kotevska](**KOTEVSKAO@ORNL.GOV**) -->


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

Here’s how to use the Gaussian mechanism in just a few lines:

```python
import numpy as np
from PETINA import DP_Mechanisms

domain = [1, 2, 13, 4, 5, 11, 21, 3, 14, 5, 10, 12, 4, 16, 7, 18, 10, 30, 20, 15, 27]
epsilon = 0.1
delta = 1e-5

print("DP Output =", DP_Mechanisms.applyDPGaussian(domain, delta, epsilon))
```
We also provide hands-on [examples](./PETINA/examples/) in the examples folder.

- [Example 1](./PETINA/examples/tutorial1_basic.py): Basic PETINA Usage.
- [Example 2](./PETINA/examples/tutorial2_CountSketch_PureLDP.py): 


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
- Oliver Kotevska – KOTEVSKAO@ORNL.GOV – Project Lead - Maintainer
- Trong Nguyen – NT9@ORNL.GOV – Intern Student


We welcome community contributions to PETINA.

For major changes, please open an issue first. For small fixes or enhancements, submit a pull request. Include/update tests where applicable.

Contact: KOTEVSKAO@ORNL.GOV

## License
This project is licensed under the MIT License.

## Acknowledgements
This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research under Contract No. DE-AC05-00OR22725. This manuscript has been co-authored by UT-Battelle, LLC under Contract No. DE-AC05-00OR22725 with the U.S. Department of Energy. The United States Government retains and the publisher, by accepting the article for publication, acknowledges that the United States Government retains a non-exclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this manuscript, or allow others to do so, for United States Government purposes. The Department of Energy will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan (http://energy.gov/downloads/doe-public-access-plan).

