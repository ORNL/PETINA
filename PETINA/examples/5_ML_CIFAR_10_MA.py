import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from PETINA import BudgetAccountant, BudgetError, DP_Mechanisms
import numpy as np
import random
import time

# --- Set seeds for reproducibility ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed=42
set_seed(seed)

# --- Setup device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# --- Load CIFAR-10 dataset ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

batch_size = 1024
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2, pin_memory=True)

# --- Simple CNN Model ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
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
def apply_laplace_with_budget(grad, sensitivity, epsilon, gamma, accountant):
    grad_np = grad.cpu().numpy()
    noisy = DP_Mechanisms.applyDPLaplace(grad_np, sensitivity=sensitivity, epsilon=epsilon, gamma=gamma, accountant=accountant)
    return torch.tensor(noisy, dtype=torch.float32).to(device)

def apply_gaussian_with_budget(grad, delta, epsilon, gamma, accountant):
    grad_np = grad.cpu().numpy()
    noisy = DP_Mechanisms.applyDPGaussian(grad_np, delta=delta, epsilon=epsilon, gamma=gamma, accountant=accountant)
    return torch.tensor(noisy, dtype=torch.float32).to(device)

# --- Training with DP and budget accounting + mixed precision ---
def train_model_with_budget(dp_type, dp_params, total_epsilon, total_delta, rounds=2, epochs_per_round=3):
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    accountant = BudgetAccountant(epsilon=total_epsilon, delta=total_delta)
    print(f"Initialized BudgetAccountant: ε={total_epsilon}, δ={total_delta}")

    try:
        for r in range(rounds):
            print(f"\nRound {r + 1}/{rounds}")
            for e in range(epochs_per_round):
                model.train()
                progress_bar = tqdm(trainloader, desc=f"Epoch {e + 1}", leave=False)
                for inputs, targets in progress_bar:
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    optimizer.zero_grad()
                    with torch.amp.autocast(device_type=device.type):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    scaler.scale(loss).backward()

                    if dp_type is not None:
                        scaler.unscale_(optimizer)
                        for p in model.parameters():
                            if p.grad is None:
                                continue
                            if dp_type == 'laplace':
                                p.grad = apply_laplace_with_budget(
                                    p.grad,
                                    sensitivity=dp_params.get('sensitivity', 1.0),
                                    epsilon=dp_params.get('epsilon', 1.0),
                                    gamma=dp_params.get('gamma', 1.0),
                                    accountant=accountant
                                )
                            elif dp_type == 'gaussian':
                                p.grad = apply_gaussian_with_budget(
                                    p.grad,
                                    delta=dp_params.get('delta', 1e-5),
                                    epsilon=dp_params.get('epsilon', 1.0),
                                    gamma=dp_params.get('gamma', 1.0),
                                    accountant=accountant
                                )
                            else:
                                raise ValueError(f"Unknown dp_type: {dp_type}")

                    scaler.step(optimizer)
                    scaler.update()

                    eps_rem, _ = accountant.remaining()
                    progress_bar.set_postfix(loss=loss.item(), eps_rem=eps_rem)

                acc = evaluate(model, testloader)
                eps_used, delta_used = accountant.total()
                eps_rem, delta_rem = accountant.remaining()

                print(f" Epoch {e + 1} Test Accuracy: {acc:.4f}")
                print(f"   Used ε: {eps_used}, δ: {delta_used}")
                print(f"   Remaining ε: {eps_rem}, δ: {delta_rem}")

                if eps_rem <= 0 and delta_rem <= 0:
                    print("\nBudget exhausted! Stopping training early.")
                    return model

    except BudgetError as be:
        print(f"\nBudgetError caught: {be}")
        print("Training stopped early due to budget exhaustion.")
    except Exception as ex:
        print(f"\nUnexpected error: {ex}")

    print("Training completed.\n")
    return model

if __name__ == "__main__":
    total_epsilon = 11000
    #Avoid using delta=1.0, as it causes remaining().delta to always be 1.0. (IBM Budget Accountant issue)
    total_delta = 1-1e-9  # Set a delta close to 1 but not exactly 1 to avoid issues with remaining budget checks
    rounds = 2
    epochs_per_round = 3
    delta=1e-5
    epsilon=1.1011632828830176
    gamma=0.01
    sensitivity = 1.0
    print("===========Parameters for DP Training===========")
    print(f"Running experiments with ε={epsilon}, δ={delta}, γ={gamma}, sensitivity={sensitivity}")
    print(f"Total rounds: {rounds}, epochs per round: {epochs_per_round}")
    print(f"Seed value for reproducibility: {seed}")
    print(f"Batch size: {batch_size}\n")


    # print("\n=== Experiment 1: No DP Noise ===")
    # train_model_with_budget(dp_type=None, dp_params={}, total_epsilon=total_epsilon, total_delta=total_delta,
    #                         rounds=rounds, epochs_per_round=epochs_per_round)

    print("\n=== Experiment 2: Gaussian DP Noise with Budget Accounting ===")
    train_model_with_budget(dp_type='gaussian',
                            dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma},
                            total_epsilon=total_epsilon, total_delta=total_delta,
                            rounds=rounds, epochs_per_round=epochs_per_round)

    print("\n=== Experiment 3: Laplace DP Noise with Budget Accounting ===")
    train_model_with_budget(dp_type='laplace',
                            dp_params={'sensitivity': sensitivity, 'epsilon': epsilon, 'gamma': gamma},
                            total_epsilon=total_epsilon, total_delta=0.0,
                            rounds=rounds, epochs_per_round=epochs_per_round)

# ---------OUTPUT--------
# Using device: cuda
# Device name: NVIDIA GH200 480GB
# ===========Parameters for DP Training===========
# Running experiments with ε=1.0, δ=1e-05, γ=0.01, sensitivity=1.0
# Total rounds: 2, epochs per round: 3
# Seed value for reproducibility: 42
# Batch size: 256


# === Experiment 1: No DP Noise ===
# Initialized BudgetAccountant: ε=11000, δ=0.999999999

# Round 1/2
#  Epoch 1 Test Accuracy: 0.4415                                                                                         
#    Used ε: 0, δ: 0.0
#    Remaining ε: 10999.99475479126, δ: 0.999999999
#  Epoch 2 Test Accuracy: 0.5214                                                                                         
#    Used ε: 0, δ: 0.0
#    Remaining ε: 10999.99475479126, δ: 0.999999999
#  Epoch 3 Test Accuracy: 0.5852                                                                                         
#    Used ε: 0, δ: 0.0
#    Remaining ε: 10999.99475479126, δ: 0.999999999

# Round 2/2
#  Epoch 1 Test Accuracy: 0.5979                                                                                         
#    Used ε: 0, δ: 0.0
#    Remaining ε: 10999.99475479126, δ: 0.999999999
#  Epoch 2 Test Accuracy: 0.6448                                                                                         
#    Used ε: 0, δ: 0.0
#    Remaining ε: 10999.99475479126, δ: 0.999999999
#  Epoch 3 Test Accuracy: 0.6679                                                                                         
#    Used ε: 0, δ: 0.0
#    Remaining ε: 10999.99475479126, δ: 0.999999999
# Training completed.


# === Experiment 2: Gaussian DP Noise with Budget Accounting ===
# Initialized BudgetAccountant: ε=11000, δ=0.999999999

# Round 1/2
#  Epoch 1 Test Accuracy: 0.4352                                                                                         
#    Used ε: 1568.0, δ: 0.015557785990390495
#    Remaining ε: 9432.002544403076, δ: 0.9999999989841963
#  Epoch 2 Test Accuracy: 0.4945                                                                                         
#    Used ε: 3136.0, δ: 0.030873527275858192
#    Remaining ε: 7863.999843597412, δ: 0.999999998968143
#  Epoch 3 Test Accuracy: 0.4971                                                                                         
#    Used ε: 4704.0, δ: 0.04595098953612253
#    Remaining ε: 6295.997142791748, δ: 0.9999999989518359

# Round 2/2
#  Epoch 1 Test Accuracy: 0.5020                                                                                         
#    Used ε: 6272.0, δ: 0.06079387986526329
#    Remaining ε: 4728.004932403564, δ: 0.999999998935271
#  Epoch 2 Test Accuracy: 0.5265                                                                                         
#    Used ε: 7840.0, δ: 0.07540584768318462
#    Remaining ε: 3160.0022315979004, δ: 0.9999999989184444
#  Epoch 3 Test Accuracy: 0.5030                                                                                         
#    Used ε: 9408.0, δ: 0.08979048563289593
#    Remaining ε: 1591.9995307922363, δ: 0.9999999989013518
# Training completed.


# === Experiment 3: Laplace DP Noise with Budget Accounting ===
# Initialized BudgetAccountant: ε=11000, δ=0.0

# Round 1/2
#  Epoch 1 Test Accuracy: 0.4376                                                                                         
#    Used ε: 1568.0, δ: 0.0
#    Remaining ε: 9432.002544403076, δ: 0.0
#  Epoch 2 Test Accuracy: 0.5160                                                                                         
#    Used ε: 3136.0, δ: 0.0
#    Remaining ε: 7863.999843597412, δ: 0.0
#  Epoch 3 Test Accuracy: 0.5770                                                                                         
#    Used ε: 4704.0, δ: 0.0
#    Remaining ε: 6295.997142791748, δ: 0.0

# Round 2/2
#  Epoch 1 Test Accuracy: 0.6017                                                                                         
#    Used ε: 6272.0, δ: 0.0
#    Remaining ε: 4728.004932403564, δ: 0.0
#  Epoch 2 Test Accuracy: 0.6242                                                                                         
#    Used ε: 7840.0, δ: 0.0
#    Remaining ε: 3160.0022315979004, δ: 0.0
#  Epoch 3 Test Accuracy: 0.6306                                                                                         
#    Used ε: 9408.0, δ: 0.0
#    Remaining ε: 1591.9995307922363, δ: 0.0
# Training completed.
