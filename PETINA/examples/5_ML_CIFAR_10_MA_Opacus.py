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
from PETINA.package.Opacus_budget_accountant.accountants import create_accountant
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
    total_epoch = 5
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


# ===========Parameters for DP Training===========
# Running experiments with ε=1, δ=1e-05, γ=0.01, sensitivity=1.0
# Epochs: 5
# Batch size: 240


# === Experiment 1: No DP Noise ===
# /mnt/c/Users/ducnguyen/Desktop/ORNL/Working_PETINA/PETINA/PETINA/package/Opacus_budget_accountant/accountants/gdp.py:23: UserWarning: GDP accounting is experimental and can underestimate privacy expenditure.Proceed with caution. More details: https://arxiv.org/pdf/2106.02848.pdf      
#   warnings.warn(
# 100%|████████████████████████████████████████████████████████| 250/250 [00:06<00:00, 39.19it/s]
# Train Epoch: 1  Loss: 0.477132 Test Accuracy = 96.73% )
# 100%|████████████████████████████████████████████████████████| 250/250 [00:07<00:00, 34.79it/s]
# Train Epoch: 2  Loss: 0.099253 Test Accuracy = 97.73% )
# 100%|████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 69.88it/s]
# Train Epoch: 3  Loss: 0.061180 Test Accuracy = 98.16% )
# 100%|████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 68.68it/s]
# Train Epoch: 4  Loss: 0.046758 Test Accuracy = 98.19% )
# 100%|████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 70.11it/s]
# Train Epoch: 5  Loss: 0.039491 Test Accuracy = 98.53% )
# Time run: 27.84 seconds


# === Experiment 2: Gaussian DP Noise with Budget Accounting ===
# 100%|████████████████████████████████████████████████████████| 250/250 [00:11<00:00, 21.53it/s]
# Train Epoch: 1  Loss: 0.895086 (ε_accountant = 0.41, δ = 1e-05 Test Accuracy = 94.32% )
# 100%|████████████████████████████████████████████████████████| 250/250 [00:15<00:00, 16.49it/s]
# Train Epoch: 2  Loss: 0.169962 (ε_accountant = 0.60, δ = 1e-05 Test Accuracy = 96.39% )
# 100%|████████████████████████████████████████████████████████| 250/250 [00:11<00:00, 21.52it/s]
# Train Epoch: 3  Loss: 0.123647 (ε_accountant = 0.75, δ = 1e-05 Test Accuracy = 96.71% )
# 100%|████████████████████████████████████████████████████████| 250/250 [00:12<00:00, 20.43it/s]
# Train Epoch: 4  Loss: 0.101407 (ε_accountant = 0.88, δ = 1e-05 Test Accuracy = 97.61% )
# 100%|████████████████████████████████████████████████████████| 250/250 [00:15<00:00, 15.83it/s]
# Train Epoch: 5  Loss: 0.091556 (ε_accountant = 1.00, δ = 1e-05 Test Accuracy = 97.32% )
# Time run: 69.43 seconds


# === Experiment 3: CSVec + Gaussian DP with Budget Accounting (r=5, c=260, blocks=1) ===        
# 100%|████████████████████████████████████████████████████████| 250/250 [00:14<00:00, 17.56it/s]
# Train Epoch: 1  Loss: 0.926059 (ε_accountant = 0.41, δ = 1e-05 Test Accuracy = 87.22% )
# 100%|████████████████████████████████████████████████████████| 250/250 [00:17<00:00, 14.60it/s]
# Train Epoch: 2  Loss: 0.333673 (ε_accountant = 0.60, δ = 1e-05 Test Accuracy = 92.84% )
# 100%|████████████████████████████████████████████████████████| 250/250 [00:13<00:00, 18.40it/s]
# Train Epoch: 3  Loss: 0.229118 (ε_accountant = 0.75, δ = 1e-05 Test Accuracy = 93.60% )
# 100%|████████████████████████████████████████████████████████| 250/250 [00:17<00:00, 14.58it/s]
# Train Epoch: 4  Loss: 0.189926 (ε_accountant = 0.88, δ = 1e-05 Test Accuracy = 94.81% )
# 100%|████████████████████████████████████████████████████████| 250/250 [00:13<00:00, 18.26it/s]
# Train Epoch: 5  Loss: 0.168853 (ε_accountant = 1.00, δ = 1e-05 Test Accuracy = 95.46% )
# Time run: 78.80 seconds