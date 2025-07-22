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
    total_epoch = 15
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
    sketch_cols = 260
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
# Epochs: 15
# Batch size: 240


# === Experiment 1: No DP Noise ===

# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:06<00:00, 38.41it/s]
# Train Epoch: 1  Loss: 0.475377 Test Accuracy = 97.27% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 61.37it/s]
# Train Epoch: 2  Loss: 0.091906 Test Accuracy = 97.70% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:07<00:00, 34.34it/s]
# Train Epoch: 3  Loss: 0.058676 Test Accuracy = 98.72% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 59.76it/s]
# Train Epoch: 4  Loss: 0.044533 Test Accuracy = 98.77% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 60.21it/s]
# Train Epoch: 5  Loss: 0.035888 Test Accuracy = 98.93% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 59.77it/s]
# Train Epoch: 6  Loss: 0.028468 Test Accuracy = 98.97% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 58.01it/s]
# Train Epoch: 7  Loss: 0.025573 Test Accuracy = 99.01% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 59.37it/s]
# Train Epoch: 8  Loss: 0.021843 Test Accuracy = 98.73% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:07<00:00, 33.89it/s]
# Train Epoch: 9  Loss: 0.018509 Test Accuracy = 99.01% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 57.06it/s]
# Train Epoch: 10         Loss: 0.015797 Test Accuracy = 99.17% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 58.88it/s]
# Train Epoch: 11         Loss: 0.013007 Test Accuracy = 98.95% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 58.88it/s]
# Train Epoch: 12         Loss: 0.010677 Test Accuracy = 99.10% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 58.48it/s]
# Train Epoch: 13         Loss: 0.010241 Test Accuracy = 98.97% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 56.41it/s]
# Train Epoch: 14         Loss: 0.009958 Test Accuracy = 99.08% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:04<00:00, 56.85it/s]
# Train Epoch: 15         Loss: 0.008137 Test Accuracy = 99.03% )
# Time run: 86.42 seconds


# === Experiment 2: Gaussian DP Noise with Budget Accounting ===
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:09<00:00, 27.20it/s]
# Train Epoch: 1  Loss: 0.819454 (ε_accountant = 0.23, δ = 1e-05 Test Accuracy = 93.43% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:09<00:00, 26.99it/s]
# Train Epoch: 2  Loss: 0.177047 (ε_accountant = 0.33, δ = 1e-05 Test Accuracy = 96.37% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:09<00:00, 27.12it/s]
# Train Epoch: 3  Loss: 0.120755 (ε_accountant = 0.41, δ = 1e-05 Test Accuracy = 97.33% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:12<00:00, 20.02it/s]
# Train Epoch: 4  Loss: 0.100059 (ε_accountant = 0.48, δ = 1e-05 Test Accuracy = 96.82% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:09<00:00, 27.31it/s]
# Train Epoch: 5  Loss: 0.089449 (ε_accountant = 0.54, δ = 1e-05 Test Accuracy = 97.74% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:09<00:00, 27.33it/s]
# Train Epoch: 6  Loss: 0.080570 (ε_accountant = 0.60, δ = 1e-05 Test Accuracy = 96.96% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:12<00:00, 20.34it/s]
# Train Epoch: 7  Loss: 0.075968 (ε_accountant = 0.65, δ = 1e-05 Test Accuracy = 97.97% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:09<00:00, 27.20it/s]
# Train Epoch: 8  Loss: 0.070158 (ε_accountant = 0.70, δ = 1e-05 Test Accuracy = 98.15% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:09<00:00, 25.46it/s]
# Train Epoch: 9  Loss: 0.068838 (ε_accountant = 0.75, δ = 1e-05 Test Accuracy = 97.81% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:12<00:00, 20.32it/s]
# Train Epoch: 10         Loss: 0.066132 (ε_accountant = 0.79, δ = 1e-05 Test Accuracy = 98.19% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:09<00:00, 26.26it/s]
# Train Epoch: 11         Loss: 0.062575 (ε_accountant = 0.84, δ = 1e-05 Test Accuracy = 97.52% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:09<00:00, 26.03it/s]
# Train Epoch: 12         Loss: 0.060539 (ε_accountant = 0.88, δ = 1e-05 Test Accuracy = 98.26% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:12<00:00, 19.61it/s]
# Train Epoch: 13         Loss: 0.058936 (ε_accountant = 0.92, δ = 1e-05 Test Accuracy = 97.97% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:09<00:00, 25.35it/s]
# Train Epoch: 14         Loss: 0.060142 (ε_accountant = 0.96, δ = 1e-05 Test Accuracy = 97.97% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:09<00:00, 26.72it/s]
# Train Epoch: 15         Loss: 0.058792 (ε_accountant = 0.99, δ = 1e-05 Test Accuracy = 97.97% )
# Time run: 164.50 seconds


# === Experiment 3: CSVec + Gaussian DP with Budget Accounting (r=5, c=260, blocks=1) ===
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:13<00:00, 18.30it/s]
# Train Epoch: 1  Loss: 1.013669 (ε_accountant = 0.23, δ = 1e-05 Test Accuracy = 84.93% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:10<00:00, 24.14it/s]
# Train Epoch: 2  Loss: 0.352913 (ε_accountant = 0.33, δ = 1e-05 Test Accuracy = 91.23% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:10<00:00, 24.08it/s]
# Train Epoch: 3  Loss: 0.245494 (ε_accountant = 0.41, δ = 1e-05 Test Accuracy = 93.69% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:13<00:00, 18.03it/s]
# Train Epoch: 4  Loss: 0.203575 (ε_accountant = 0.48, δ = 1e-05 Test Accuracy = 93.95% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:10<00:00, 24.08it/s]
# Train Epoch: 5  Loss: 0.181024 (ε_accountant = 0.54, δ = 1e-05 Test Accuracy = 95.36% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:10<00:00, 23.26it/s]
# Train Epoch: 6  Loss: 0.160444 (ε_accountant = 0.60, δ = 1e-05 Test Accuracy = 95.67% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:13<00:00, 18.11it/s]
# Train Epoch: 7  Loss: 0.151420 (ε_accountant = 0.65, δ = 1e-05 Test Accuracy = 95.89% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:09<00:00, 25.21it/s]
# Train Epoch: 8  Loss: 0.144431 (ε_accountant = 0.70, δ = 1e-05 Test Accuracy = 95.66% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:10<00:00, 23.51it/s]
# Train Epoch: 9  Loss: 0.135495 (ε_accountant = 0.75, δ = 1e-05 Test Accuracy = 96.18% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:12<00:00, 20.10it/s]
# Train Epoch: 10         Loss: 0.127903 (ε_accountant = 0.79, δ = 1e-05 Test Accuracy = 95.94% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:09<00:00, 26.35it/s]
# Train Epoch: 11         Loss: 0.124706 (ε_accountant = 0.84, δ = 1e-05 Test Accuracy = 96.73% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:09<00:00, 26.94it/s]
# Train Epoch: 12         Loss: 0.122459 (ε_accountant = 0.88, δ = 1e-05 Test Accuracy = 95.83% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:12<00:00, 19.89it/s]
# Train Epoch: 13         Loss: 0.118744 (ε_accountant = 0.92, δ = 1e-05 Test Accuracy = 96.48% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:09<00:00, 26.61it/s]
# Train Epoch: 14         Loss: 0.115895 (ε_accountant = 0.96, δ = 1e-05 Test Accuracy = 96.68% )
# 100%|████████████████████████████████████████████████████████████████████████████| 250/250 [00:09<00:00, 25.39it/s]
# Train Epoch: 15         Loss: 0.112523 (ε_accountant = 0.99, δ = 1e-05 Test Accuracy = 95.96% )
# Time run: 177.19 seconds