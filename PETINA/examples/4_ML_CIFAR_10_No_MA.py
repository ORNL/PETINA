# File: PETINA/PETINA/examples/4_ML_CIFAR_10_No_MA.py
# ======================================================
#        CIFAR-10 Training with Differential Privacy
# ======================================================
import math, random, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# --- PETINA (Differential Privacy Toolkit) Imports ---
from PETINA import DP_Mechanisms
from PETINA.Data_Conversion_Helper import TypeConverter
from PETINA.package.csvec.csvec import CSVec


# =========================
# 1. Setup and Utilities
# =========================
def set_seed(seed=42):
    """Ensure reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"Device name: {torch.cuda.get_device_name(0)}")
# =========================
# 2. Load CIFAR-10 Dataset
# =========================
batch_size = 256

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# =========================
# 3. Define Simple CNN
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# =========================
# 4. Evaluation Function
# =========================
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

# =========================
# 5. DP Noise Functions
# =========================
def no_noise(grad, **kwargs):
    return grad

def gaussian_noise(grad, delta, epsilon, gamma):
    return DP_Mechanisms.applyDPGaussian(grad, delta, epsilon, gamma)

def laplace_noise(grad, sensitivity, epsilon, gamma):
    return DP_Mechanisms.applyDPLaplace(grad, sensitivity, epsilon, gamma)

# =========================
# 6. Training Function
# =========================
def train_model(dp_noise=None, dp_params=None, rounds=2, epochs_per_round=3,
                use_count_sketch=False, sketch_params=None):
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    for r in range(rounds):
        print(f"\nRound {r + 1}/{rounds}")
        for e in range(epochs_per_round):
            model.train()
            progress = tqdm(trainloader, desc=f"Epoch {e + 1}", leave=False)
            for inputs, targets in progress:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()

                # Flatten gradients into 1D vector
                grad_list = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
                if not grad_list: continue
                flat_grad = torch.cat(grad_list)

                # --- Apply DP Mechanism ---
                if use_count_sketch:
                    privatized = DP_Mechanisms.applyCountSketch(
                        domain=flat_grad,
                        sketch_rows=sketch_params['d'],
                        sketch_cols=sketch_params['w'],
                        dp_mechanism=dp_noise,
                        dp_params=dp_params,
                        num_blocks=sketch_params.get('numBlocks', 1),
                        device=device
                    )
                    # Restore back to model
                    idx = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            numel = p.grad.numel()
                            p.grad = privatized[idx:idx + numel].view_as(p.grad).clone().detach().to(device)
                            idx += numel
                elif dp_noise:
                    for p in model.parameters():
                        if p.grad is not None:
                            noisy = dp_noise(p.grad.cpu().numpy(), **dp_params)
                            p.grad = torch.tensor(noisy, dtype=torch.float32).to(device)

                scaler.step(optimizer)
                scaler.update()
                progress.set_postfix(loss=loss.item())

            acc = evaluate(model, testloader)
            print(f" Epoch {e + 1} Accuracy: {acc:.4f}")
    print("Training Done.")
    return model

# =========================
# 7. Experiment Settings
# =========================
delta       = 1e-5
epsilon     = 1.0
gamma       = 0.01
sensitivity = 1.0
rounds      = 2
epochs      = 3
sketch_rows = 5       # hash functions (d)
sketch_cols = 10000   # sketch width (w)
csvec_blocks = 1

print("===== Differential Privacy Parameters =====")
print(f"ε={epsilon}, δ={delta}, γ={gamma}, sensitivity={sensitivity}")
print(f"Using Count Sketch rows={sketch_rows}, cols={sketch_cols}, blocks={csvec_blocks}")
print("===========================================\n")

# =========================
# 8. Run Experiments with Timing
# =========================

# === No DP ===
print("=== No DP Noise ===")
start = time.time()
train_model(dp_noise=no_noise, dp_params={}, rounds=rounds, epochs_per_round=epochs)
print(f"Time run: {time.time() - start:.2f} seconds\n")

# === Gaussian DP ===
print("=== Gaussian DP Noise ===")
start = time.time()
train_model(dp_noise=gaussian_noise,
            dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma},
            rounds=rounds, epochs_per_round=epochs)
print(f"Time run: {time.time() - start:.2f} seconds\n")

# === Laplace DP ===
print("=== Laplace DP Noise ===")
start = time.time()
train_model(dp_noise=laplace_noise,
            dp_params={'sensitivity': sensitivity, 'epsilon': epsilon, 'gamma': gamma},
            rounds=rounds, epochs_per_round=epochs)
print(f"Time run: {time.time() - start:.2f} seconds\n")

# === CSVec + Gaussian DP ===
print("=== CSVec + Gaussian DP ===")
start = time.time()
train_model(dp_noise=gaussian_noise,
            dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma},
            rounds=rounds, epochs_per_round=epochs,
            use_count_sketch=True,
            sketch_params={'d': sketch_rows, 'w': sketch_cols, 'numBlocks': csvec_blocks})
print(f"Time run: {time.time() - start:.2f} seconds\n")

# === CSVec + Laplace DP ===
print("=== CSVec + Laplace DP ===")
start = time.time()
train_model(dp_noise=laplace_noise,
            dp_params={'sensitivity': sensitivity, 'epsilon': epsilon, 'gamma': gamma},
            rounds=rounds, epochs_per_round=epochs,
            use_count_sketch=True,
            sketch_params={'d': sketch_rows, 'w': sketch_cols, 'numBlocks': csvec_blocks})
print(f"Time run: {time.time() - start:.2f} seconds\n")



# =========================OUTPUT=========================
# Using device: cuda
# Device name: NVIDIA GeForce RTX 3060 Ti
# ===== Differential Privacy Parameters =====
# ε=1.0, δ=1e-05, γ=0.01, sensitivity=1.0
# Using Count Sketch rows=5, cols=10000, blocks=1
# ===========================================

# === No DP Noise ===

# Round 1/2
#  Epoch 1 Accuracy: 0.4423                                                                         
#  Epoch 2 Accuracy: 0.5210                                                                         
#  Epoch 3 Accuracy: 0.5843                                                                         

# Round 2/2
#  Epoch 1 Accuracy: 0.5981                                                                         
#  Epoch 2 Accuracy: 0.6443                                                                         
#  Epoch 3 Accuracy: 0.6637                                                                         
# Training Done.
# Time run: 32.16 seconds

# === Gaussian DP Noise ===

# Round 1/2
#  Epoch 1 Accuracy: 0.4399                                                                         
#  Epoch 2 Accuracy: 0.5065                                                                         
#  Epoch 3 Accuracy: 0.5612                                                                         

# Round 2/2
#  Epoch 1 Accuracy: 0.6129                                                                         
#  Epoch 2 Accuracy: 0.6385                                                                         
#  Epoch 3 Accuracy: 0.6759                                                                         
# Training Done.
# Time run: 183.59 seconds

# === Laplace DP Noise ===

# Round 1/2
#  Epoch 1 Accuracy: 0.4314                                                                         
#  Epoch 2 Accuracy: 0.5185                                                                         
#  Epoch 3 Accuracy: 0.5849                                                                         

# Round 2/2
#  Epoch 1 Accuracy: 0.6050                                                                         
#  Epoch 2 Accuracy: 0.6444                                                                         
#  Epoch 3 Accuracy: 0.6555                                                                         
# Training Done.
# Time run: 183.64 seconds

# === CSVec + Gaussian DP ===

# Round 1/2
#  Epoch 1 Accuracy: 0.4458                                                                         
#  Epoch 2 Accuracy: 0.5152                                                                         
#  Epoch 3 Accuracy: 0.5690                                                                         

# Round 2/2
#  Epoch 1 Accuracy: 0.5861                                                                         
#  Epoch 2 Accuracy: 0.5981                                                                         
#  Epoch 3 Accuracy: 0.6363                                                                         
# Training Done.
# Time run: 277.68 seconds

# === CSVec + Laplace DP ===

# Round 1/2
#  Epoch 1 Accuracy: 0.4477                                                                         
#  Epoch 2 Accuracy: 0.5141                                                                         
#  Epoch 3 Accuracy: 0.5702                                                                         

# Round 2/2
#  Epoch 1 Accuracy: 0.6107                                                                         
#  Epoch 2 Accuracy: 0.6261                                                                         
#  Epoch 3 Accuracy: 0.6399                                                                         
# Training Done.
# Time run: 280.79 seconds