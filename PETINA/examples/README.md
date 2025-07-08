# PETINA Examples

This repository contains several examples to help you practice and explore the **PETINA** library.

## Getting Started

To set up the environment, install the required dependencies from [requirements.txt](requirements.txt):

```bash
pip install -r requirements.txt
```
## Example List
### [Example 1: Basic PETINA Usage](tutorial1_basic.py)

This script demonstrates how to use core features of the PETINA library, including:

- Generating synthetic data
- Applying DP mechanisms: Laplace, Gaussian, Exponential, SVT
- Encoding techniques: Unary and Histogram
- Clipping and Pruning (fixed/adaptive)
- Computing helper values like `p`, `q`, `gamma`, and `sigma`

Useful for getting a quick hands-on overview of PETINA’s building blocks.

### [Example 2: Frequency Estimation with PETINA CMS and CSVec](tutorial2_CountSketch_PureLDP.py)
This script is adapted from the pure-LDP library and extends it with:
- PETINA CMS: Count Mean Sketch for LDP data using PETINA
- Centralized CMS: PETINA's centralized Count Sketch alternative
- CSVec: PETINA’s implementation of Count Sketch Vector for high-dimensional privatized data
- A simulation of Prefix Extending Method (PEM) for heavy hitter discovery
Key features demonstrated:
- Compare frequency estimation error (variance)** across the following methods:
  - OUE, OLH, THE, HR (pure LDP)
  - Apple CMS (LDP)
  - PETINA CMS (LDP)
  - PETINA Centralized CMS
  - PETINA CSVec

- Evaluate heavy hitters using the Prefix Extending Method (PEM)
