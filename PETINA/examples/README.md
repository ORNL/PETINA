# PETINA Examples

This repository contains several examples to help you practice and explore the **PETINA** library.

## Getting Started

To set up the environment, install the required dependencies from [requirements.txt](requirements.txt):

```bash
pip install -r requirements.txt
```

## Example List
### [Example 1: Basic PETINA Usage](1_basic.py)
This script demonstrates how to use core features of the PETINA library, including:
- Generating synthetic data
- Applying DP mechanisms: Laplace, Gaussian, Exponential, SVT
- Encoding techniques: Unary and Histogram
- Clipping and Pruning (fixed/adaptive)
- Computing helper values like `p`, `q`, `gamma`, and `sigma`
- Useful for getting a quick hands-on overview of PETINA’s building blocks.


### [Example 2: PETINA with Real-World Data](2_Personal_data.py)
This script demonstrates applying PETINA’s differential privacy techniques on a real-world dataset (UCI Adult dataset):
- Handles categorical data (education) with unary encoding combined with randomized response to privately estimate category counts.
- Applies the Laplace mechanism to numerical data (age) for privacy-preserving noise addition.
- Shows clipping to limit large numerical values and pruning to remove small values from the dataset.
- Illustrates practical DP applications on mixed-type real data, useful for privacy-preserving data analysis.
- Good for understanding how PETINA can protect real datasets combining categorical and numerical features.

### [Example 3: PETINA with Iris Dataset](3_Iris_data.py)
This script showcases PETINA’s differential privacy techniques applied to the Iris dataset:
- Uses unary encoding with DP randomized response on the categorical target variable (species) to privately estimate class counts.
- Applies the Laplace mechanism to add noise to each numerical feature (sepal length, sepal width, petal length, petal width) for privacy.
- Demonstrates adaptive clipping on a numerical feature to limit extreme values while preserving data utility.
- Great for learning how PETINA handles mixed data types in a classic ML dataset with DP protections.

### [Example 4: MNIST Training with Differential Privacy and Count Sketch - No Budget Accounting](4_ML_MNIST_No_MA.py)
This example demonstrates training a simple CNN on MNIST with PETINA’s differential privacy mechanisms, including:
* Standard training without DP noise (baseline)
* Gaussian and Laplace DP noise injection on gradients
* Count Sketch (CSVec) compression combined with DP noise to reduce gradient size and improve privacy-utility tradeoff
* Support for both Gaussian and Laplace noise with and without sketching
* Real-time training progress with tqdm and accuracy evaluation after each epoch
* Configurable sketch parameters (rows, cols, blocks) and DP parameters (ε, δ, γ, sensitivity)
* Illustrates practical usage of PETINA’s DP mechanisms and Count Sketch utility in a centralized supervised learning pipeline
### [Example 5: DP Training with Count Sketch Compression on MNIST - With Budget Accounting](5_ML_MNIST_MA_Opacus.py)
This example demonstrates training a CNN on MNIST with differential privacy (DP) using PETINA, featuring:
- Basic CNN architecture implemented in PyTorch
- Support for DP noise injection using Gaussian mechanism with budget accounting (via Opacus GDP accountant)
- Integration of Count Sketch (CSVec) compression combined with DP noise to reduce gradient dimensionality and communication cost
- Three experimental modes:
    1. Training without DP noise (baseline)
    2. Training with Gaussian DP noise and privacy budget tracking
    3. Training with Count Sketch + Gaussian DP noise for compressed privatized gradients
- Evaluation on MNIST test set after each epoch
- Illustrates practical usage of PETINA’s DP mechanisms and sketching utilities in centralized training pipelines

### [Example 6: Federated Learning with DP and Count Sketch - With Budget Accounting](6_FL_MNIST_MA_Opacus.py)
This example showcases a federated learning pipeline on MNIST using PETINA, featuring:
- Federated client-server architecture with multiple clients
- Local training of CNN models on client data with optional DP noise (Gaussian)
- Count Sketch (CSVec) compression of model updates combined with DP for communication efficiency
- Privacy budget accounting using Opacus’ GDP accountant
- Aggregation of privatized updates and global model evaluation over multiple rounds
- Demonstrates practical integration of PETINA’s DP mechanisms in federated settings with advanced sketching