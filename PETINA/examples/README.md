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
- Useful for getting a quick hands-on overview of PETINA’s building blocks.


### [Example 2: PETINA with Real-World Data](tutorial2_Personal_data.py)
This script demonstrates applying PETINA’s differential privacy techniques on a real-world dataset (UCI Adult dataset):
- Handles categorical data (education) with unary encoding combined with randomized response to privately estimate category counts.
- Applies the Laplace mechanism to numerical data (age) for privacy-preserving noise addition.
- Shows clipping to limit large numerical values and pruning to remove small values from the dataset.
- Illustrates practical DP applications on mixed-type real data, useful for privacy-preserving data analysis.
- Good for understanding how PETINA can protect real datasets combining categorical and numerical features.

### [Example 3: PETINA with Iris Dataset](tutorial3_Iris_data.py)
This script showcases PETINA’s differential privacy techniques applied to the Iris dataset:
- Uses unary encoding with DP randomized response on the categorical target variable (species) to privately estimate class counts.
- Applies the Laplace mechanism to add noise to each numerical feature (sepal length, sepal width, petal length, petal width) for privacy.
- Demonstrates adaptive clipping on a numerical feature to limit extreme values while preserving data utility.
- Great for learning how PETINA handles mixed data types in a classic ML dataset with DP protections.

### [Example 4: Training CNN on CIFAR-10 with PETINA Differential Privacy](tutorial4_ML_CIFAR_10_No_MA.py)
This example demonstrates training a Convolutional Neural Network (CNN) on the CIFAR-10 dataset with differentially private gradient updates using PETINA:
- Implements DP-SGD style training by adding noise to model gradients each batch.
- Supports three experiments:
- No privacy — standard training without noise.
- Gaussian noise — adds Gaussian noise calibrated to privacy parameters.
- Laplace noise — adds Laplace noise calibrated to privacy parameters.
- Allows configurable privacy parameters (epsilon, delta, gamma, sensitivity).
- Measures and prints training time for each experiment.
- Evaluates test accuracy after each epoch to observe impact of privacy noise on model performance.
- Great for understanding the tradeoff between privacy and model accuracy when training deep learning models with PETINA.