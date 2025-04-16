# PETINA: Privacy prEservaTIoN Algorithms
PETINA is a Python package with a commonly and state of the art differential privacy (DP) algorithms, with applications to both supervised and unsupervised learning tasks and decimal and categorical data types. Most common packages a limited in diversity of DP algorithms or hard to implement into the existing code.

If you use any of the code please use the appropriate citation from here https://www.osti.gov/doecode/biblio/149859.

## Summary
This package includes functions for:
- Applying DPGaussian, RDPGaussian, DPExponential, DPLaplace, unaryEncoding, histogramEncoding, DPClipping, DPPruning, DPPercentale, DP Sparse Vector mechanisms using custom noise scaling
- Applying CountSketch and FastProjUnit sketching based mechanisms 
- Adaptive clipping and pruning mechanisms
- Calculating the p and q based on given epsilon value
- Help functions for converting list to torch, torch to list, list to numpy and numpy to list
- Utility functions for type conversion

## Installation
Or install from source:
```bash
git clone https://github.com/ORNL/PETINA.git
cd petina
pip install -e .
```

You can install the package directly from PyPI:
```bash
pip install petina
```

## Quick Start
Here's a simple example of how to use PETINA:
```python
import numpy as np
from petina import baselines

domain = [1, 2, 13, 4, 5, 11, 21, 3, 14, 5, 10, 12, 4, 16, 7, 18, 10, 30, 20, 15, 27]
epsilon = 0.1
delta = 10e-5

print("DP = ", baselines.applyDPGaussian(domain, delta, epsilon))
```

## Acknowledgement
This manuscript has been co-authored by UT-Battelle, LLC, under Contract No. DE-AC05-00OR22725 with the U.S. Department of Energy. The United States Government retains and the publisher, by accepting the article for publication, acknowledges that the United States Government retains a nonexclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this manuscript, or allow others to do so, for United States Government purposes. The Department of Energy will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan (http://energy.gov/downloads/doe-public-access-plan).
