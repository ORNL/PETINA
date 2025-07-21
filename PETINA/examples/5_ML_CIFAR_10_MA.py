import math
import random
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import warnings # Required for the IBM BudgetAccountant's internal warnings
from numbers import Real, Integral # Required for check_epsilon_delta and BudgetAccountant
from PETINA.Data_Conversion_Helper import TypeConverter
from PETINA.package.csvec.csvec import CSVec
from PETINA import BudgetAccountant, BudgetError

class DP_Mechanisms:
    @staticmethod
    def applyDPGaussian(domain: np.ndarray, delta: float = 1e-5, epsilon: float = 0.1, gamma: float = 1.0) -> np.ndarray:
        """
        Applies Gaussian noise to the input NumPy array for differential privacy,
        and optionally tracks budget via a BudgetAccountant.
        This function expects and returns a NumPy array.
        """
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * gamma / epsilon
        privatized = domain + np.random.normal(loc=0, scale=sigma, size=domain.shape) * 1.572 # Retaining *1.572 from your original code
        
        # if accountant is not None:
        #     accountant.spend(epsilon, delta)
        return privatized

    @staticmethod
    def applyDPLaplace(domain: np.ndarray, sensitivity: float = 1, epsilon: float = 0.01, gamma: float = 1) -> np.ndarray:
        """
        Applies Laplace noise to the input NumPy array for differential privacy.
        Tracks privacy budget with an optional BudgetAccountant.
        This function expects and returns a NumPy array.
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be > 0 for Laplace mechanism.")
        scale = sensitivity * gamma / epsilon
        privatized = domain + np.random.laplace(loc=0, scale=scale, size=domain.shape)

        # if accountant is not None:
        #     cost_epsilon, cost_delta = epsilon, 0.0 # Laplace mechanism typically has delta=0
        #     accountant.spend(cost_epsilon, cost_delta)
            
        return privatized

    @staticmethod
    def applyCountSketch(
        domain: list | np.ndarray | torch.Tensor,
        num_rows: int,
        num_cols: int,
        epsilon: float,
        delta: float,
        mechanism: str = "gaussian",
        sensitivity: float = 1.0,
        gamma: float = 0.01,
        num_blocks: int = 1,
        device: torch.device | str | None = None
    ) -> list | np.ndarray | torch.Tensor:
        """
        Applies Count Sketch to the input data, then adds differential privacy
        noise to the sketched representation, and finally reconstructs the data.
        Consumes budget from the provided BudgetAccountant.
        """
        converter = TypeConverter(domain)
        flattened_data_tensor, original_shape = converter.get()

        # Ensure tensor
        if not isinstance(flattened_data_tensor, torch.Tensor):
            flattened_data_tensor = torch.tensor(flattened_data_tensor, dtype=torch.float32)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)

        flattened_data_tensor = flattened_data_tensor.to(device)

        csvec_instance = CSVec(
            d=flattened_data_tensor.numel(),
            c=num_cols,
            r=num_rows,
            numBlocks=num_blocks,
            device=device
        )

        csvec_instance.accumulateVec(flattened_data_tensor)

        sketched_table_np = csvec_instance.table.detach().cpu().numpy()

        if mechanism == "gaussian":
            noisy_sketched_table_np = DP_Mechanisms.applyDPGaussian(
                sketched_table_np, delta=delta, epsilon=epsilon, gamma=gamma
            )
        elif mechanism == "laplace":
            noisy_sketched_table_np = DP_Mechanisms.applyDPLaplace(
                sketched_table_np, sensitivity=sensitivity, epsilon=epsilon, gamma=gamma
            )
        else:
            raise ValueError(f"Unsupported DP mechanism for Count Sketch: {mechanism}. Choose 'gaussian' or 'laplace'.")

        csvec_instance.table = torch.tensor(noisy_sketched_table_np, dtype=torch.float32).to(device)
        reconstructed_noisy_data = csvec_instance._findAllValues()

        return converter.restore(reconstructed_noisy_data.tolist())


# File: PETINA/PETINA/examples/4_ML_CIFAR_10_No_MA.py
# ======================================================
#         CIFAR-10 Training with Differential Privacy
# ======================================================

# --- Set seeds for reproducibility ---
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# seed=42
# set_seed(seed)

# --- Setup device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# --- Load CIFAR-10 dataset ---
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# batch_size = 1024 
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
# testloader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# --- Load MNIST dataset ---
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST normalization
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
testset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

batch_size = 240 # Or whatever you want
testbatchsize=1024
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
testloader  = torch.utils.data.DataLoader(testset, batch_size=testbatchsize, shuffle=False, num_workers=2, pin_memory=True)

# --- Simple CNN Model ---
class SimpleCNN(nn.Module):
    # def __init__(self):
    #     super().__init__()
    #     self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    #     self.fc1 = nn.Linear(64 * 8 * 8, 256)
    #     self.fc2 = nn.Linear(256, 10)
    #     self.relu = nn.ReLU()

    # def forward(self, x):
    #     x = self.pool(self.relu(self.conv1(x)))
    #     x = self.pool(self.relu(self.conv2(x)))
    #     x = x.view(-1, 64 * 8 * 8)
    #     x = self.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # Change 3 -> 1 for MNIST
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # adjust from 5*5 to 4*4
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 6, 24, 24]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 16, 8, 8] → pool → [batch, 16, 4, 4]
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
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
# These functions now correctly handle the conversion to/from NumPy for DP_Mechanisms
def apply_laplace_with_budget(grad: torch.Tensor, sensitivity: float, epsilon: float, gamma: float, accountant: BudgetAccountant) -> torch.Tensor:
    grad_np = grad.cpu().numpy() # Convert PyTorch Tensor to NumPy array
    noisy_np = DP_Mechanisms.applyDPLaplace(grad_np, sensitivity=sensitivity, epsilon=epsilon, gamma=gamma)
    return torch.tensor(noisy_np, dtype=torch.float32).to(device) # Convert NumPy array back to PyTorch Tensor

def apply_gaussian_with_budget(grad: torch.Tensor, delta: float, epsilon: float, gamma: float, accountant: BudgetAccountant) -> torch.Tensor:
    grad_np = grad.cpu().numpy() # Convert PyTorch Tensor to NumPy array
    noisy_np = DP_Mechanisms.applyDPGaussian(grad_np, delta=delta, epsilon=epsilon, gamma=gamma)
    return torch.tensor(noisy_np, dtype=torch.float32).to(device) # Convert NumPy array back to PyTorch Tensor

# --- Training with DP and budget accounting + mixed precision ---
def train_model_with_budget(dp_type, dp_params, total_epsilon, total_delta, rounds=2, epochs_per_round=3, use_count_sketch=False, sketch_params=None):
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    accountant = BudgetAccountant(epsilon=total_epsilon, delta=total_delta)
    print(f"Initialized BudgetAccountant: ε={total_epsilon}, δ={total_delta}")

    mechanism_map = {
        'gaussian': "gaussian",
        'laplace': "laplace"
    }

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
                        
                        if use_count_sketch:
                            grad_list = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
                            if not grad_list: continue
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
                        else: # Direct DP application (without Count Sketch)
                            for p in model.parameters():
                                if p.grad is None: continue
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

                    # eps_rem, _ = accountant.remaining()
                    progress_bar.set_postfix(loss=loss.item())

                acc = evaluate(model, testloader)
                if dp_type=='laplace' or use_count_sketch:
                    epsilon = epsilon=dp_params.get('epsilon', 1.0)
                    accountant.spend(epsilon=epsilon, delta=0)
                elif dp_type=='gaussian' or use_count_sketch:
                    epsilon =dp_params.get('epsilon', 1.0)
                    delta=dp_params.get('delta', 1e-5)
                    accountant.spend(epsilon=epsilon, delta=delta)
                eps_used, delta_used = accountant.total()
                eps_rem, delta_rem = accountant.remaining()

                print(f" Epoch {e + 1} Test Accuracy: {acc}")
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
    total_epsilon = 30
    # Avoid using delta=1.0, as it causes remaining().delta to always be 1.0. (IBM Budget Accountant issue)
    total_delta = 1-1e-9 # Set a delta close to 1 but not exactly 1 to avoid issues with remaining budget checks
    rounds = 5
    epochs_per_round = 5
    delta=1e-5
    epsilon= 1
    gamma=0.01
    sensitivity = 1.0
    print("===========Parameters for DP Training===========")
    print(f"Running experiments with ε={epsilon}, δ={delta}, γ={gamma}, sensitivity={sensitivity}")
    print(f"Total rounds: {rounds}, epochs per round: {epochs_per_round}")
    # print(f"Seed value for reproducibility: {seed}")
    print(f"Batch size: {batch_size}\n")


    print("\n=== Experiment 1: No DP Noise ===")
    start = time.time()
    train_model_with_budget(dp_type=None, dp_params={}, total_epsilon=total_epsilon, total_delta=total_delta,
                            rounds=rounds, epochs_per_round=epochs_per_round)
    print(f"Time run: {time.time() - start:.2f} seconds\n")

    print("\n=== Experiment 2: Gaussian DP Noise with Budget Accounting ===")
    start = time.time()
    train_model_with_budget(dp_type='gaussian',
                            dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma},
                            total_epsilon=total_epsilon, total_delta=total_delta,
                            rounds=rounds, epochs_per_round=epochs_per_round)
    print(f"Time run: {time.time() - start:.2f} seconds\n")

    print("\n=== Experiment 3: Laplace DP Noise with Budget Accounting ===")
    start = time.time()
    train_model_with_budget(dp_type='laplace',
                            dp_params={'sensitivity': sensitivity, 'epsilon': epsilon, 'gamma': gamma},
                            total_epsilon=total_epsilon, total_delta=0.0, # Delta is typically 0 for pure Laplace
                            rounds=rounds, epochs_per_round=epochs_per_round)
    print(f"Time run: {time.time() - start:.2f} seconds\n")
    Count Sketch parameters
    sketch_rows = 5
    sketch_cols = 10000
    csvec_blocks = 1
    sketch_rows = 3
    sketch_cols = 2048
    csvec_blocks = 1
    print(f"\n=== Experiment 4: CSVec + Gaussian DP with Budget Accounting (r={sketch_rows}, c={sketch_cols}, blocks={csvec_blocks}) ===")
    start = time.time()
    train_model_with_budget(dp_type='gaussian',
                            dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma, 'sensitivity': sensitivity}, # Pass sensitivity for CSVec context
                            total_epsilon=total_epsilon, total_delta=total_delta,
                            rounds=rounds, epochs_per_round=epochs_per_round,
                            use_count_sketch=True,
                            sketch_params={'d': sketch_rows, 'w': sketch_cols, 'numBlocks': csvec_blocks})
    print(f"Time run: {time.time() - start:.2f} seconds\n")
    
    print(f"\n=== Experiment 5: CSVec + Laplace DP with Budget Accounting (r={sketch_rows}, c={sketch_cols}, blocks={csvec_blocks}) ===")
    start = time.time()
    train_model_with_budget(dp_type='laplace',
                            dp_params={'delta': delta,'sensitivity': sensitivity, 'epsilon': epsilon, 'gamma': gamma},
                            total_epsilon=total_epsilon, total_delta=0.0, # Delta is typically 0 for pure Laplace
                            rounds=rounds, epochs_per_round=epochs_per_round,
                            use_count_sketch=True,
                            sketch_params={'d': sketch_rows, 'w': sketch_cols, 'numBlocks': csvec_blocks})
    print(f"Time run: {time.time() - start:.2f} seconds\n")
# ----------------OUTPUT-------------------
# Using device: cuda
# Device name: NVIDIA GeForce RTX 3060 Ti
# ===========Parameters for DP Training===========
# Running experiments with ε=1.1011632828830176, δ=1e-05, γ=0.01, sensitivity=1.0
# Total rounds: 2, epochs per round: 3
# Seed value for reproducibility: 42
# Batch size: 1024


# === Experiment 1: No DP Noise ===
# Initialized BudgetAccountant: ε=11000, δ=0.999999999

# Round 1/2
#  Epoch 1 Test Accuracy: 0.2733                                                                    
#    Used ε: 0, δ: 0.0
#    Remaining ε: 10999.99475479126, δ: 0.999999999
#  Epoch 2 Test Accuracy: 0.3528                                                                    
#    Used ε: 0, δ: 0.0
#    Remaining ε: 10999.99475479126, δ: 0.999999999
#  Epoch 3 Test Accuracy: 0.4188                                                                    
#    Used ε: 0, δ: 0.0
#    Remaining ε: 10999.99475479126, δ: 0.999999999

# Round 2/2
#  Epoch 1 Test Accuracy: 0.4632                                                                    
#    Used ε: 0, δ: 0.0
#    Remaining ε: 10999.99475479126, δ: 0.999999999
#  Epoch 2 Test Accuracy: 0.4941                                                                    
#    Used ε: 0, δ: 0.0
#    Remaining ε: 10999.99475479126, δ: 0.999999999
#  Epoch 3 Test Accuracy: 0.5176                                                                    
#    Used ε: 0, δ: 0.0
#    Remaining ε: 10999.99475479126, δ: 0.999999999
# Training completed.

# Time run: 33.21 seconds


# === Experiment 2: Gaussian DP Noise with Budget Accounting ===
# Initialized BudgetAccountant: ε=11000, δ=0.999999999

# Round 1/2
#  Epoch 1 Test Accuracy: 0.2883                                                                    
#    Used ε: 431.6560068901441, δ: 0.003912346352998807
#    Remaining ε: 10568.345546722412, δ: 0.9999999989960723
#  Epoch 2 Test Accuracy: 0.379                                                                     
#    Used ε: 863.3120137802886, δ: 0.007809386252011792
#    Remaining ε: 10136.685848236084, δ: 0.9999999989921292
#  Epoch 3 Test Accuracy: 0.4266                                                                    
#    Used ε: 1294.968020670405, δ: 0.01169117958118839
#    Remaining ε: 9705.036640167236, δ: 0.9999999989881706

# Round 2/2
#  Epoch 1 Test Accuracy: 0.4451                                                                    
#    Used ε: 1726.6240275605048, δ: 0.015557785990390495
#    Remaining ε: 9273.376941680908, δ: 0.9999999989841963
#  Epoch 2 Test Accuracy: 0.4642                                                                    
#    Used ε: 2158.2800344506277, δ: 0.019409264896109064
#    Remaining ε: 8841.71724319458, δ: 0.9999999989802066
#  Epoch 3 Test Accuracy: 0.479                                                                     
#    Used ε: 2589.9360413408167, δ: 0.02324567548237717
#    Remaining ε: 8410.068035125732, δ: 0.9999999989762012
# Training completed.

# Time run: 52.04 seconds


# === Experiment 3: Laplace DP Noise with Budget Accounting ===
# Initialized BudgetAccountant: ε=11000, δ=0.0

# Round 1/2
#  Epoch 1 Test Accuracy: 0.2865                                                                    
#    Used ε: 431.6560068901441, δ: 0.0
#    Remaining ε: 10568.345546722412, δ: 0.0
#  Epoch 2 Test Accuracy: 0.3836                                                                    
#    Used ε: 863.3120137802886, δ: 0.0
#    Remaining ε: 10136.685848236084, δ: 0.0
#  Epoch 3 Test Accuracy: 0.442                                                                     
#    Used ε: 1294.968020670405, δ: 0.0
#    Remaining ε: 9705.036640167236, δ: 0.0

# Round 2/2
#  Epoch 1 Test Accuracy: 0.4816                                                                    
#    Used ε: 1726.6240275605048, δ: 0.0
#    Remaining ε: 9273.376941680908, δ: 0.0
#  Epoch 2 Test Accuracy: 0.483                                                                     
#    Used ε: 2158.2800344506277, δ: 0.0
#    Remaining ε: 8841.71724319458, δ: 0.0
#  Epoch 3 Test Accuracy: 0.5251                                                                    
#    Used ε: 2589.9360413408167, δ: 0.0
#    Remaining ε: 8410.068035125732, δ: 0.0
# Training completed.

# Time run: 48.41 seconds


# === Experiment 4: CSVec + Gaussian DP with Budget Accounting (r=5, c=10000, blocks=1) ===
# Initialized BudgetAccountant: ε=11000, δ=0.999999999

# Round 1/2
#  Epoch 1 Test Accuracy: 0.2817                                                                    
#    Used ε: 53.95700086126781, δ: 0.0004898824184218814
#    Remaining ε: 10946.042537689209, δ: 0.9999999989995099
#  Epoch 2 Test Accuracy: 0.3948                                                                    
#    Used ε: 107.9140017225358, δ: 0.0009795248520598839
#    Remaining ε: 10892.090320587158, δ: 0.9999999989990196
#  Epoch 3 Test Accuracy: 0.4377                                                                    
#    Used ε: 161.87100258380386, δ: 0.0014689274184783345
#    Remaining ε: 10838.127613067627, δ: 0.999999998998529

# Round 2/2
#  Epoch 1 Test Accuracy: 0.4748                                                                    
#    Used ε: 215.82800344507191, δ: 0.0019580902351839656
#    Remaining ε: 10784.175395965576, δ: 0.999999998998038
#  Epoch 2 Test Accuracy: 0.4945                                                                    
#    Used ε: 269.78500430633994, δ: 0.0024470134196259478
#    Remaining ε: 10730.212688446045, δ: 0.999999998997547
#  Epoch 3 Test Accuracy: 0.5109                                                                    
#    Used ε: 323.742005167608, δ: 0.002935697089195911
#    Remaining ε: 10676.260471343994, δ: 0.9999999989970557
# Training completed.

# Time run: 38.91 seconds


# === Experiment 5: CSVec + Laplace DP with Budget Accounting (r=5, c=10000, blocks=1) ===
# Initialized BudgetAccountant: ε=11000, δ=0.0

# Round 1/2
#  Epoch 1 Test Accuracy: 0.2691                                                                    
#    Used ε: 53.95700086126781, δ: 0.0
#    Remaining ε: 10946.042537689209, δ: 0.0
#  Epoch 2 Test Accuracy: 0.3625                                                                    
#    Used ε: 107.9140017225358, δ: 0.0
#    Remaining ε: 10892.090320587158, δ: 0.0
#  Epoch 3 Test Accuracy: 0.4078                                                                    
#    Used ε: 161.87100258380386, δ: 0.0
#    Remaining ε: 10838.127613067627, δ: 0.0

# Round 2/2
#  Epoch 1 Test Accuracy: 0.4647                                                                    
#    Used ε: 215.82800344507191, δ: 0.0
#    Remaining ε: 10784.175395965576, δ: 0.0
#  Epoch 2 Test Accuracy: 0.4825                                                                    
#    Used ε: 269.78500430633994, δ: 0.0
#    Remaining ε: 10730.212688446045, δ: 0.0
#  Epoch 3 Test Accuracy: 0.499                                                                     
#    Used ε: 323.742005167608, δ: 0.0
#    Remaining ε: 10676.260471343994, δ: 0.0
# Training completed.

# Time run: 38.76 seconds