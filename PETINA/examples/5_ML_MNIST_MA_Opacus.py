# === Standard Libraries ===
import time

# === Third-Party Libraries ===
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# === Opacus Libraries ===
from PETINA.package.Opacus_budget_accountant.accountants.gdp import GaussianAccountant
from PETINA.package.Opacus_budget_accountant.accountants.utils import get_noise_multiplier
# === PETINA Modules ===
from PETINA import  DP_Mechanisms


# --- Load MNIST dataset ---
# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
batch_size = 240 
testbatchsize=1024
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)) # Standard MNIST normalization
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
testset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
dataset_size = len(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
testloader  = torch.utils.data.DataLoader(testset, batch_size=testbatchsize, shuffle=False, num_workers=2, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model ----------
class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) 
        x = F.max_pool2d(x, 2, 1)  
        x = F.relu(self.conv2(x)) 
        x = F.max_pool2d(x, 2, 1)  
        x = x.view(-1, 32 * 4 * 4) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        return x
# --- Evaluation ---
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    return correct / total

# --- DP noise wrappers with budget accounting ---
def apply_gaussian_with_budget(grad: torch.Tensor, delta: float, epsilon: float, gamma: float) -> torch.Tensor:
    grad_np = grad.cpu().numpy() # Convert PyTorch Tensor to NumPy array
    noisy_np = DP_Mechanisms.applyDPGaussian(grad_np, delta=delta, epsilon=epsilon, gamma=gamma)
    return torch.tensor(noisy_np, dtype=torch.float32).to(device) # Convert NumPy array back to PyTorch Tensor
def getModelDimension(model):
    params = [p.detach().view(-1) for p in model.parameters()]  # Flatten each parameter
    flat_tensor = torch.cat(params)  # Concatenate into a single 1D tensor
    return len(flat_tensor)


# --- Training with DP and budget accounting + mixed precision ---
def train_model_with_budget(dp_type, dp_params,total_epoch=5, use_count_sketch=False, sketch_params=None):
    model = SampleConvNet().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.25, momentum=0)
    accountantOPC = GaussianAccountant()
    mechanism_map = {
        'gaussian': "gaussian"
    }
    for e in range(total_epoch):
        model.train()
        criterion = nn.CrossEntropyLoss()
        losses = []

        for _batch_idx, (data, target) in enumerate(tqdm(trainloader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()

            if dp_type is not None:
                if use_count_sketch:
                    grad_list = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
                    if not grad_list:
                        continue
                    flat_grad = torch.cat(grad_list)

                    mechanism_str = mechanism_map.get(dp_type)
                    if mechanism_str is None:
                        raise ValueError(f"Unsupported DP noise type '{dp_type}' for Count Sketch DP.")

                    privatized_grad_tensor = DP_Mechanisms.applyCountSketch(
                        domain=flat_grad,
                        num_rows=sketch_params['d'],
                        num_cols=sketch_params['w'],
                        epsilon=dp_params['epsilon'],
                        delta=dp_params['delta'],
                        mechanism=mechanism_str,
                        sensitivity=dp_params.get('sensitivity', 1.0),
                        gamma=dp_params.get('gamma', 0.01),
                        num_blocks=sketch_params.get('numBlocks', 1),
                        device=device
                    )
                    
                    idx = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            numel = p.grad.numel()
                            grad_slice = privatized_grad_tensor[idx:idx + numel]
                            p.grad = grad_slice.detach().clone().view_as(p.grad).to(device)
                            idx += numel
                else:
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        if dp_type == 'gaussian':
                            p.grad = apply_gaussian_with_budget(
                                p.grad,
                                delta=dp_params.get('delta', 1e-5),
                                epsilon=dp_params.get('epsilon', 1.0),
                                gamma=dp_params.get('gamma', 1.0)
                            )
                        else:
                            raise ValueError(f"Unknown dp_type: {dp_type}")

                sample_rate = trainloader.batch_size / dataset_size            
                sigma = get_noise_multiplier(
                    target_epsilon=dp_params['epsilon'],
                    target_delta=dp_params['delta'],
                    sample_rate=sample_rate,
                    epochs=total_epoch,
                    accountant="gdp",
                )
                accountantOPC.step(noise_multiplier=sigma, sample_rate=sample_rate)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
    
        acc = evaluate(model, testloader)
        loss_str = f"Train Epoch: {e+1} \tLoss: {np.mean(losses):.6f}"
        if dp_type is not None:
            epsilon = accountantOPC.get_epsilon(delta=dp_params.get('delta', 1e-5))
            print(f"{loss_str} (ε_accountant = {epsilon:.2f}, δ = {dp_params.get('delta', 1e-5)} Test Accuracy = {acc * 100:.2f}% )")
        else:
            print(f"{loss_str} Test Accuracy = {acc * 100:.2f}% )")

def main():
    total_epoch = 20
    delta=1e-5
    epsilon= 1
    gamma=0.01
    sensitivity = 1.0

    print("===========Parameters for DP Training===========")
    print(f"Running experiments with ε={epsilon}, δ={delta}, γ={gamma}, sensitivity={sensitivity}")
    print(f"Epochs: {total_epoch}")
    print(f"Batch size: {batch_size}\n")


    print("\n=== Experiment 1: No DP Noise ===")
    start = time.time()
    train_model_with_budget(dp_type=None, dp_params={},total_epoch=total_epoch)
    print(f"Time run: {time.time() - start:.2f} seconds\n")

    print("\n=== Experiment 2: Gaussian DP Noise with Budget Accounting ===")
    start = time.time()
    train_model_with_budget(dp_type='gaussian',
                            dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma},
                            total_epoch=total_epoch)
    print(f"Time run: {time.time() - start:.2f} seconds\n")
    
    sketch_rows = 5
    sketch_cols = 520 #x10
    csvec_blocks = 1
    print(f"\n=== Experiment 3: CSVec + Gaussian DP with Budget Accounting (r={sketch_rows}, c={sketch_cols}, blocks={csvec_blocks}) ===")
    start = time.time()
    train_model_with_budget(dp_type='gaussian',
                            dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma, 'sensitivity': sensitivity}, 
                            total_epoch=total_epoch,
                            use_count_sketch=True,
                            sketch_params={'d': sketch_rows, 'w': sketch_cols, 'numBlocks': csvec_blocks})
    print(f"Time run: {time.time() - start:.2f} seconds\n")

if __name__ == "__main__":
    main()

# --------------OUTPUT--------------
# ===========Parameters for DP Training===========
# Running experiments with ε=1, δ=1e-05, γ=0.01, sensitivity=1.0
# Epochs: 20
# Batch size: 240


# === Experiment 1: No DP Noise ===
# /autofs/nccs-svm1_envoy_od/nt9/summer_2025/improving_petina/work_Intern/PETINA/PETINA/package/Opacus_budget_accountant/accountants/gdp.py:23: UserWarning: GDP accounting is experimental and can underestimate privacy expenditure.Proceed with caution. More details: https://arxiv.org/pdf/2106.02848.pdf
#   warnings.warn(
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 81.79it/s]
# Train Epoch: 1  Loss: 0.439115 Test Accuracy = 96.62% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:02<00:00, 85.74it/s]
# Train Epoch: 2  Loss: 0.090492 Test Accuracy = 98.30% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:02<00:00, 86.72it/s]
# Train Epoch: 3  Loss: 0.060656 Test Accuracy = 98.64% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:02<00:00, 86.22it/s]
# Train Epoch: 4  Loss: 0.046135 Test Accuracy = 97.97% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:02<00:00, 86.18it/s]
# Train Epoch: 5  Loss: 0.034723 Test Accuracy = 98.81% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:02<00:00, 86.86it/s]
# Train Epoch: 6  Loss: 0.031744 Test Accuracy = 98.93% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:02<00:00, 85.55it/s]
# Train Epoch: 7  Loss: 0.026989 Test Accuracy = 98.58% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:02<00:00, 86.09it/s]
# Train Epoch: 8  Loss: 0.021672 Test Accuracy = 99.13% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:02<00:00, 85.71it/s]
# Train Epoch: 9  Loss: 0.018428 Test Accuracy = 98.94% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:02<00:00, 86.90it/s]
# Train Epoch: 10         Loss: 0.015423 Test Accuracy = 99.03% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:02<00:00, 86.12it/s]
# Train Epoch: 11         Loss: 0.014802 Test Accuracy = 98.92% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:02<00:00, 85.61it/s]
# Train Epoch: 12         Loss: 0.015109 Test Accuracy = 99.09% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:02<00:00, 86.60it/s]
# Train Epoch: 13         Loss: 0.010979 Test Accuracy = 98.99% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:02<00:00, 85.18it/s]
# Train Epoch: 14         Loss: 0.010142 Test Accuracy = 98.57% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:02<00:00, 85.89it/s]
# Train Epoch: 15         Loss: 0.009143 Test Accuracy = 99.09% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:02<00:00, 86.14it/s]
# Train Epoch: 16         Loss: 0.009388 Test Accuracy = 99.09% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:02<00:00, 86.71it/s]
# Train Epoch: 17         Loss: 0.007797 Test Accuracy = 99.13% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:02<00:00, 86.32it/s]
# Train Epoch: 18         Loss: 0.006685 Test Accuracy = 99.19% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:02<00:00, 86.05it/s]
# Train Epoch: 19         Loss: 0.004876 Test Accuracy = 99.00% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:02<00:00, 86.15it/s]
# Train Epoch: 20         Loss: 0.003585 Test Accuracy = 99.21% )
# Time run: 68.39 seconds


# === Experiment 2: Gaussian DP Noise with Budget Accounting ===
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 82.35it/s]
# Train Epoch: 1  Loss: 1.022430 (ε_accountant = 0.19, δ = 1e-05 Test Accuracy = 92.17% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 82.84it/s]
# Train Epoch: 2  Loss: 0.177818 (ε_accountant = 0.28, δ = 1e-05 Test Accuracy = 95.26% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 83.10it/s]
# Train Epoch: 3  Loss: 0.126233 (ε_accountant = 0.35, δ = 1e-05 Test Accuracy = 96.84% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 82.75it/s]
# Train Epoch: 4  Loss: 0.104728 (ε_accountant = 0.41, δ = 1e-05 Test Accuracy = 97.48% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 82.73it/s]
# Train Epoch: 5  Loss: 0.088662 (ε_accountant = 0.47, δ = 1e-05 Test Accuracy = 97.15% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 83.09it/s]
# Train Epoch: 6  Loss: 0.081958 (ε_accountant = 0.52, δ = 1e-05 Test Accuracy = 97.98% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 82.76it/s]
# Train Epoch: 7  Loss: 0.076861 (ε_accountant = 0.56, δ = 1e-05 Test Accuracy = 96.58% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 81.94it/s]
# Train Epoch: 8  Loss: 0.073504 (ε_accountant = 0.61, δ = 1e-05 Test Accuracy = 97.85% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 82.76it/s]
# Train Epoch: 9  Loss: 0.070089 (ε_accountant = 0.65, δ = 1e-05 Test Accuracy = 97.98% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 81.88it/s]
# Train Epoch: 10         Loss: 0.067998 (ε_accountant = 0.68, δ = 1e-05 Test Accuracy = 98.02% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 82.26it/s]
# Train Epoch: 11         Loss: 0.067034 (ε_accountant = 0.72, δ = 1e-05 Test Accuracy = 98.24% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 82.93it/s]
# Train Epoch: 12         Loss: 0.063648 (ε_accountant = 0.76, δ = 1e-05 Test Accuracy = 98.04% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 82.10it/s]
# Train Epoch: 13         Loss: 0.063215 (ε_accountant = 0.79, δ = 1e-05 Test Accuracy = 98.34% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 80.54it/s]
# Train Epoch: 14         Loss: 0.063938 (ε_accountant = 0.82, δ = 1e-05 Test Accuracy = 98.37% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 81.04it/s]
# Train Epoch: 15         Loss: 0.061167 (ε_accountant = 0.85, δ = 1e-05 Test Accuracy = 98.17% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 78.13it/s]
# Train Epoch: 16         Loss: 0.058835 (ε_accountant = 0.88, δ = 1e-05 Test Accuracy = 98.28% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 82.92it/s]
# Train Epoch: 17         Loss: 0.060573 (ε_accountant = 0.91, δ = 1e-05 Test Accuracy = 98.25% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:02<00:00, 83.72it/s]
# Train Epoch: 18         Loss: 0.059899 (ε_accountant = 0.94, δ = 1e-05 Test Accuracy = 98.20% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 83.04it/s]
# Train Epoch: 19         Loss: 0.058437 (ε_accountant = 0.97, δ = 1e-05 Test Accuracy = 98.08% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 81.71it/s]
# Train Epoch: 20         Loss: 0.057776 (ε_accountant = 1.00, δ = 1e-05 Test Accuracy = 98.13% )
# Time run: 71.16 seconds


# === Experiment 3: CSVec + Gaussian DP with Budget Accounting (r=5, c=520, blocks=1) ===
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 60.81it/s]
# Train Epoch: 1  Loss: 0.803630 (ε_accountant = 0.19, δ = 1e-05 Test Accuracy = 91.33% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 60.48it/s]
# Train Epoch: 2  Loss: 0.229522 (ε_accountant = 0.28, δ = 1e-05 Test Accuracy = 95.12% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 61.47it/s]
# Train Epoch: 3  Loss: 0.161331 (ε_accountant = 0.35, δ = 1e-05 Test Accuracy = 95.21% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 61.00it/s]
# Train Epoch: 4  Loss: 0.133786 (ε_accountant = 0.41, δ = 1e-05 Test Accuracy = 96.25% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 60.01it/s]
# Train Epoch: 5  Loss: 0.115497 (ε_accountant = 0.47, δ = 1e-05 Test Accuracy = 97.06% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 60.90it/s]
# Train Epoch: 6  Loss: 0.106055 (ε_accountant = 0.52, δ = 1e-05 Test Accuracy = 96.84% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 60.73it/s]
# Train Epoch: 7  Loss: 0.098079 (ε_accountant = 0.56, δ = 1e-05 Test Accuracy = 97.07% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 60.59it/s]
# Train Epoch: 8  Loss: 0.092462 (ε_accountant = 0.61, δ = 1e-05 Test Accuracy = 97.42% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 60.59it/s]
# Train Epoch: 9  Loss: 0.087937 (ε_accountant = 0.65, δ = 1e-05 Test Accuracy = 97.25% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 60.79it/s]
# Train Epoch: 10         Loss: 0.084235 (ε_accountant = 0.68, δ = 1e-05 Test Accuracy = 97.23% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 61.02it/s]
# Train Epoch: 11         Loss: 0.079506 (ε_accountant = 0.72, δ = 1e-05 Test Accuracy = 97.58% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 60.55it/s]
# Train Epoch: 12         Loss: 0.079485 (ε_accountant = 0.76, δ = 1e-05 Test Accuracy = 97.51% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 60.81it/s]
# Train Epoch: 13         Loss: 0.077530 (ε_accountant = 0.79, δ = 1e-05 Test Accuracy = 97.79% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 60.74it/s]
# Train Epoch: 14         Loss: 0.073882 (ε_accountant = 0.82, δ = 1e-05 Test Accuracy = 97.39% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 59.25it/s]
# Train Epoch: 15         Loss: 0.072434 (ε_accountant = 0.85, δ = 1e-05 Test Accuracy = 97.57% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 59.30it/s]
# Train Epoch: 16         Loss: 0.072189 (ε_accountant = 0.88, δ = 1e-05 Test Accuracy = 97.81% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 59.13it/s]
# Train Epoch: 17         Loss: 0.069862 (ε_accountant = 0.91, δ = 1e-05 Test Accuracy = 97.80% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 59.18it/s]
# Train Epoch: 18         Loss: 0.067649 (ε_accountant = 0.94, δ = 1e-05 Test Accuracy = 97.86% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 59.46it/s]
# Train Epoch: 19         Loss: 0.069621 (ε_accountant = 0.97, δ = 1e-05 Test Accuracy = 97.82% )
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 59.21it/s]
# Train Epoch: 20         Loss: 0.066276 (ε_accountant = 1.00, δ = 1e-05 Test Accuracy = 97.75% )
# Time run: 92.97 seconds