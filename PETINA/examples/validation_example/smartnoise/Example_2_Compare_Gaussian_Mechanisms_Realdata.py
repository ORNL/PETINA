import os
import numpy as np
import opendp.prelude as dp
from tqdm import trange
from PETINA import DP_Mechanisms
from sklearn.datasets import load_iris
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

dp.enable_features("contrib")

# --- Configuration ---
epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]  # Privacy budgets for trade-off curve
num_trials = 10000  # Reduce trials for speed
datasets_info = {}

# ----------------------------
# Utility Functions
# ----------------------------
def run_trials(true_value, sensitivity, epsilon, label):
    # Gaussian scale: usually scale = sensitivity * sqrt(2*log(1.25/delta)) / epsilon
    # For simplicity, fix delta=1e-5 (typical small value)
    delta = 1e-5
    import math
    scale = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    input_space = dp.vector_domain(dp.atom_domain(T=float))
    metric = dp.l2_distance(T=float)  # Gaussian mechanism uses L2 distance

    gaussian_mech = dp.m.make_gaussian(input_space, metric, scale)

    noisy_opendp_results = []
    noisy_petina_results = []

    data = [float(true_value)]

    for _ in trange(num_trials, desc=f"{label} ε={epsilon}: Running trials"):
        noisy_opendp = gaussian_mech(data)
        noisy_opendp_results.append(noisy_opendp[0])

        # Assuming PETINA has Gaussian noise methodomain, delta=10e-5, epsilon=1, gamma=1
        noisy_petina = DP_Mechanisms.applyDPGaussian(
            data,
            epsilon=epsilon,
            gamma=sensitivity,
            delta=delta
        )
        noisy_petina_results.append(noisy_petina[0])

    return np.array(noisy_opendp_results), np.array(noisy_petina_results)

def summarize(results, true_value):
    mean = results.mean()
    std = results.std()
    bias = mean - true_value
    mse = ((results - true_value) ** 2).mean()
    return mean, std, bias, mse

def plot_histogram(opendp_results, petina_results, true_value, label, epsilon):
    plt.figure(figsize=(8,5))
    bins = 50
    plt.hist(opendp_results, bins=bins, alpha=0.6, label='OpenDP (SmartNoise)', color='blue', density=True)
    plt.hist(petina_results, bins=bins, alpha=0.6, label='PETINA', color='orange', density=True)
    plt.axvline(true_value, color='green', linestyle='dashed', linewidth=2, label='True Value')
    plt.title(f"{label} Gaussian Noisy Outputs Distribution\nε={epsilon}")
    plt.xlabel("Noisy Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    os.makedirs("Plot/Example_2", exist_ok=True)
    filename = os.path.join("Plot","Example_2", f"{label.lower().replace(' ', '_')}_epsilon_{epsilon}_histogram.png")
    plt.savefig(filename)
    plt.close()

def plot_bar_errors(stats_dict, label):
    labels = list(stats_dict.keys())
    means = [stats_dict[l]['mse'] for l in labels]
    stds = [stats_dict[l]['std'] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8,5))
    rects1 = ax.bar(x - width/2, means, width, label='MSE')
    rects2 = ax.bar(x + width/2, stds, width, label='Std Dev')

    ax.set_ylabel('Error')
    ax.set_title(f'Error metrics by ε for {label}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True)

    os.makedirs("Plot/Example_2", exist_ok=True)
    filename = os.path.join("Plot","Example_2", f"{label.lower().replace(' ', '_')}_error_metrics.png")
    plt.savefig(filename)
    plt.close()

def plot_privacy_utility(epsilons, opendp_mses, petina_mses, label):
    plt.figure(figsize=(8,5))
    plt.plot(epsilons, opendp_mses, marker='o', label='OpenDP MSE')
    plt.plot(epsilons, petina_mses, marker='o', label='PETINA MSE')
    plt.xlabel('Privacy Budget ε')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(f'Privacy-Utility Trade-off for {label}')
    plt.legend()
    plt.grid(True)
    os.makedirs("Plot/Example_2", exist_ok=True)
    filename = os.path.join("Plot","Example_2", f"{label.lower().replace(' ', '_')}_privacy_utility_tradeoff.png")
    plt.savefig(filename)
    plt.close()

def plot_boxplot(opendp_results, petina_results, true_value, label, epsilon):
    plt.figure(figsize=(6,5))
    data = [opendp_results - true_value, petina_results - true_value]
    plt.boxplot(data, labels=['OpenDP Error', 'PETINA Error'])
    plt.title(f"{label} Error Distribution (ε={epsilon})")
    plt.ylabel('Error (Noisy value - True value)')
    plt.grid(True)
    os.makedirs("Plot/Example_2", exist_ok=True)
    filename = os.path.join("Plot","Example_2", f"{label.lower().replace(' ', '_')}_epsilon_{epsilon}_error_boxplot.png")
    plt.savefig(filename)
    plt.close()

# ----------------------------
# Prepare datasets
# ----------------------------
transform = transforms.Compose([transforms.ToTensor()])

mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
images = [mnist_data[i][0].numpy().flatten() for i in range(100)]
mnist_avg = np.mean([img.mean() for img in images])
sensitivity_mnist = 1.0 / 100

cifar_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
cifar_images = [cifar_data[i][0].numpy().flatten() for i in range(100)]
cifar_avg = np.mean([img.mean() for img in cifar_images])
sensitivity_cifar = 1.0 / 100

iris = load_iris()
petal_lengths = iris.data[:, 2]
iris_sample = np.random.choice(petal_lengths, size=30, replace=False)
iris_avg = np.mean(iris_sample)
sensitivity_iris = (iris_sample.max() - iris_sample.min()) / 30

datasets_info = {
    "MNIST": (mnist_avg, sensitivity_mnist),
    "CIFAR-10": (cifar_avg, sensitivity_cifar),
    "Iris": (iris_avg, sensitivity_iris),
}

# ----------------------------
# Run for all epsilons and datasets
# ----------------------------
for label, (true_val, sens) in datasets_info.items():
    print(f"\n\n=== Dataset: {label} ===")
    opendp_mse_list = []
    petina_mse_list = []
    error_stats = {}

    for eps in epsilons:
        opendp_res, petina_res = run_trials(true_val, sens, eps, label)
        mean_o, std_o, bias_o, mse_o = summarize(opendp_res, true_val)
        mean_p, std_p, bias_p, mse_p = summarize(petina_res, true_val)

        print(f"ε={eps:.2f} OpenDP: Mean={mean_o:.4f}, Std={std_o:.4f}, Bias={bias_o:.4f}, MSE={mse_o:.6f}")
        print(f"ε={eps:.2f} PETINA: Mean={mean_p:.4f}, Std={std_p:.4f}, Bias={bias_p:.4f}, MSE={mse_p:.6f}")

        opendp_mse_list.append(mse_o)
        petina_mse_list.append(mse_p)

        error_stats[f"ε={eps}"] = {"mse": mse_p, "std": std_p}

        plot_histogram(opendp_res, petina_res, true_val, label, eps)
        plot_boxplot(opendp_res, petina_res, true_val, label, eps)

    plot_bar_errors(error_stats, label)
    plot_privacy_utility(epsilons, opendp_mse_list, petina_mse_list, label)

print("All plots saved in 'Plot/' directory.")
