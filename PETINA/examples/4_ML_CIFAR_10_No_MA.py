#This example demonstrates how to train a Convolutional Neural Network (CNN) on the CIFAR-10 dataset using differential privacy (DP) mechanisms provided by the PETINA framework. It adds Gaussian (or Laplace) noise to gradients during training to ensure privacy, simulating a DP-SGD-style approach
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from PETINA import DP_Mechanisms
import time

# --- Set random seeds for reproducibility ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed=42
set_seed(seed)

# --- Setup device ---
if torch.cuda.is_available():
    print(f"Device available: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("No GPU found. Using CPU.")
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Load CIFAR-10 dataset with larger batch size ---
batch_size = 256  # Increased batch size for better GPU utilization

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2,
                                          worker_init_fn=worker_init_fn)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# --- Simple CNN Model ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Evaluation function ---
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    return correct / total

# --- Training function with mixed precision and DP noise ---
def train_model(dp_noise=None, dp_params=None, rounds=2, epochs_per_round=3):
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    scaler = torch.amp.GradScaler('cuda')


    for r in range(rounds):
        print(f"\nRound {r + 1}/{rounds}")
        for e in range(epochs_per_round):
            model.train()
            progress_bar = tqdm(trainloader, desc=f"Epoch {e + 1}", leave=False)
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda'): # Enable mixed precision
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()

                # Apply DP noise if specified
                if dp_noise is not None:
                    for p in model.parameters():
                        if p.grad is not None:
                            grad_np = p.grad.detach().cpu().numpy()
                            noisy_grad = dp_noise(grad_np, **dp_params)
                            p.grad = torch.tensor(noisy_grad, dtype=torch.float32).to(device)

                scaler.step(optimizer)
                scaler.update()

                progress_bar.set_postfix(loss=loss.item())

            acc = evaluate(model, testloader)
            print(f" Epoch {e + 1} Test Accuracy: {acc:.4f}")
    print("Training completed.\n")
    return model

# --- Define DP noise functions wrapping PETINA methods ---
def no_noise(grad, **kwargs):
    return grad  # No noise

def gaussian_noise(grad, delta, epsilon, gamma):
    return DP_Mechanisms.applyDPGaussian(grad, delta, epsilon, gamma)

def laplace_noise(grad, sensitivity, epsilon, gamma):
    return DP_Mechanisms.applyDPLaplace(grad, sensitivity, epsilon, gamma)

# --- Experiment parameters ---
delta = 1e-5
epsilon = 1.0
gamma = 0.01
sensitivity = 1.0
rounds = 2
epochs_per_round = 3
print("===========Parameters for DP Training===========")
print(f"Running experiments with ε={epsilon}, δ={delta}, γ={gamma}, sensitivity={sensitivity}")
print(f"Total rounds: {rounds}, epochs per round: {epochs_per_round}\n")
print(f"Seed value for reproducibility: {seed}\n")
print(f"Batch size: {batch_size}\n")
print("=== Experiment 1: No Privacy ===")
start = time.time()
train_model(dp_noise=no_noise, dp_params={}, rounds=rounds, epochs_per_round=epochs_per_round)
print(f"Time run: {time.time() - start:.2f} seconds\n")

print("=== Experiment 2: Gaussian DP Noise ===")
start = time.time()
train_model(dp_noise=gaussian_noise,
            dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma},
            rounds=rounds, epochs_per_round=epochs_per_round)
print(f"Time run: {time.time() - start:.2f} seconds\n")

print("=== Experiment 3: Laplace DP Noise ===")
start = time.time()
train_model(dp_noise=laplace_noise,
            dp_params={'sensitivity': sensitivity, 'epsilon': epsilon, 'gamma': gamma},
            rounds=rounds, epochs_per_round=epochs_per_round)
print(f"Time run: {time.time() - start:.2f} seconds\n")


#--------OUTPUT--------
# Device available: NVIDIA GH200 480GB
# Using device: cuda
# ===========Parameters for DP Training===========
# Running experiments with ε=1.0, δ=1e-05, γ=0.01, sensitivity=1.0
# Total rounds: 2, epochs per round: 3

# Seed value for reproducibility: 42

# Batch size: 256

# === Experiment 1: No Privacy ===

# Round 1/2
#  Epoch 1 Test Accuracy: 0.4415                                                                                         
#  Epoch 2 Test Accuracy: 0.5214                                                                                         
#  Epoch 3 Test Accuracy: 0.5852                                                                                         

# Round 2/2
#  Epoch 1 Test Accuracy: 0.5979                                                                                         
#  Epoch 2 Test Accuracy: 0.6448                                                                                         
#  Epoch 3 Test Accuracy: 0.6679                                                                                         
# Training completed.

# Time run: 21.08 seconds

# === Experiment 2: Gaussian DP Noise ===

# Round 1/2
#  Epoch 1 Test Accuracy: 0.4402                                                                                         
#  Epoch 2 Test Accuracy: 0.5085                                                                                         
#  Epoch 3 Test Accuracy: 0.5582                                                                                         

# Round 2/2
#  Epoch 1 Test Accuracy: 0.6155                                                                                         
#  Epoch 2 Test Accuracy: 0.6347                                                                                         
#  Epoch 3 Test Accuracy: 0.6755                                                                                         
# Training completed.

# Time run: 115.18 seconds

# === Experiment 3: Laplace DP Noise ===

# Round 1/2
#  Epoch 1 Test Accuracy: 0.4303                                                                                         
#  Epoch 2 Test Accuracy: 0.5173                                                                                         
#  Epoch 3 Test Accuracy: 0.5853                                                                                         

# Round 2/2
#  Epoch 1 Test Accuracy: 0.6076                                                                                         
#  Epoch 2 Test Accuracy: 0.6490                                                                                         
#  Epoch 3 Test Accuracy: 0.6580                                                                                         
# Training completed.

# Time run: 121.67 seconds
