import math
import random
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
import warnings # Required for the IBM BudgetAccountant's internal warnings
from numbers import Real, Integral # Required for check_epsilon_delta and BudgetAccountant
from PETINA.Data_Conversion_Helper import TypeConverter
from PETINA.package.csvec.csvec import CSVec
from PETINA import BudgetAccountant, BudgetError

class DP_Mechanisms:
    @staticmethod
    def applyDPGaussian(domain: np.ndarray, delta: float = 1e-5, epsilon: float = 0.1, gamma: float = 1.0, accountant: BudgetAccountant | None = None) -> np.ndarray:
        """
        Applies Gaussian noise to the input NumPy array for differential privacy,
        and optionally tracks budget via a BudgetAccountant.
        This function expects and returns a NumPy array.
        """
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * gamma / epsilon
        privatized = domain + np.random.normal(loc=0, scale=sigma, size=domain.shape) * 1.572 # Retaining *1.572 from your original code
        
        if accountant is not None:
            accountant.spend(epsilon, delta)
        return privatized

    @staticmethod
    def applyDPLaplace(domain: np.ndarray, sensitivity: float = 1, epsilon: float = 0.01, gamma: float = 1, accountant: BudgetAccountant | None = None) -> np.ndarray:
        """
        Applies Laplace noise to the input NumPy array for differential privacy.
        Tracks privacy budget with an optional BudgetAccountant.
        This function expects and returns a NumPy array.
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be > 0 for Laplace mechanism.")
        scale = sensitivity * gamma / epsilon
        privatized = domain + np.random.laplace(loc=0, scale=scale, size=domain.shape)

        if accountant is not None:
            cost_epsilon, cost_delta = epsilon, 0.0 # Laplace mechanism typically has delta=0
            accountant.spend(cost_epsilon, cost_delta)
            
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
        device: torch.device | str | None = None,
        accountant: BudgetAccountant | None = None
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
                sketched_table_np, delta=delta, epsilon=epsilon, gamma=gamma, accountant=accountant
            )
        elif mechanism == "laplace":
            noisy_sketched_table_np = DP_Mechanisms.applyDPLaplace(
                sketched_table_np, sensitivity=sensitivity, epsilon=epsilon, gamma=gamma, accountant=accountant
            )
        else:
            raise ValueError(f"Unsupported DP mechanism for Count Sketch: {mechanism}. Choose 'gaussian' or 'laplace'.")

        csvec_instance.table = torch.tensor(noisy_sketched_table_np, dtype=torch.float32).to(device)
        reconstructed_noisy_data = csvec_instance._findAllValues()

        return converter.restore(reconstructed_noisy_data.tolist())


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

# --- Load CIFAR-10 dataset with Data Augmentation ---
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # CIFAR-10 specific normalization
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

batch_size = 128 # Smaller batch size for better generalization and faster updates
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
testloader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


class ResNet18CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.model(x)


# --- Simple CNN Model (kept for reference, but won't be used for 90% accuracy goal) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
# These functions now correctly handle the conversion to/from NumPy for DP_Mechanisms
def apply_laplace_with_budget(grad: torch.Tensor, sensitivity: float, epsilon: float, gamma: float, accountant: BudgetAccountant) -> torch.Tensor:
    grad_np = grad.cpu().numpy() # Convert PyTorch Tensor to NumPy array
    noisy_np = DP_Mechanisms.applyDPLaplace(grad_np, sensitivity=sensitivity, epsilon=epsilon, gamma=gamma, accountant=accountant)
    return torch.tensor(noisy_np, dtype=torch.float32).to(device) # Convert NumPy array back to PyTorch Tensor

def apply_gaussian_with_budget(grad: torch.Tensor, delta: float, epsilon: float, gamma: float, accountant: BudgetAccountant) -> torch.Tensor:
    grad_np = grad.cpu().numpy() # Convert PyTorch Tensor to NumPy array
    noisy_np = DP_Mechanisms.applyDPGaussian(grad_np, delta=delta, epsilon=epsilon, gamma=gamma, accountant=accountant)
    return torch.tensor(noisy_np, dtype=torch.float32).to(device) # Convert NumPy array back to PyTorch Tensor

# --- Training with DP and budget accounting + mixed precision ---
def train_model_with_budget(dp_type, dp_params, total_epsilon, total_delta, rounds=2, epochs_per_round=3, use_count_sketch=False, sketch_params=None):
    # CHANGED: Use ResNet18CIFAR10 for better accuracy
    model = ResNet18CIFAR10().to(device) 
    criterion = nn.CrossEntropyLoss()
    # CHANGED: Added weight_decay for regularization
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    accountant = BudgetAccountant(epsilon=total_epsilon, delta=total_delta)
    print(f"Initialized BudgetAccountant: ε={total_epsilon}, δ={total_delta}")

    mechanism_map = {
        'gaussian': "gaussian",
        'laplace': "laplace"
    }

    # CHANGED: Add learning rate scheduler for non-DP runs
    scheduler = None
    if dp_type is None: # Only apply scheduler for non-DP to focus on hitting 90%
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)


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
                        max_grad_norm = 1.0 

                        # Compute total gradient norm
                        total_norm = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5

                        # Clip gradients if norm exceeds max_grad_norm
                        clip_coef = max_grad_norm / (total_norm + 1e-6)
                        if clip_coef < 1:
                            for p in model.parameters():
                                if p.grad is not None:
                                    p.grad.data.mul_(clip_coef)
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
                                device=device,
                                accountant=accountant # Pass the accountant object
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

                    eps_rem, _ = accountant.remaining()
                    progress_bar.set_postfix(loss=loss.item(), eps_rem=f"{eps_rem}")

                acc = evaluate(model, testloader)
                eps_used, delta_used = accountant.total()
                eps_rem, delta_rem = accountant.remaining()

                print(f" Epoch {e + 1} Test Accuracy: {acc}")
                print(f"   Used ε: {eps_used}, δ: {delta_used}")
                print(f"   Remaining ε: {eps_rem}, δ: {delta_rem}")

                # CHANGED: Step the scheduler only for non-DP runs
                if dp_type is None and scheduler is not None:
                    scheduler.step(acc)
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"   Current LR: {current_lr:.6f}")


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
    total_epsilon = 3000
    total_delta = 1-1e-9 # Set a delta close to 1 but not exactly 1 to avoid issues with remaining budget checks
    rounds = 10 # Set to 10 rounds as per your request
    epochs_per_round = 4 # Keep this as 4, so 10 rounds * 4 epochs/round = 40 total epochs
    delta=1e-4
    epsilon= 0.001
    gamma=1e-7
    sensitivity = 1.0
    print("===========Parameters for DP Training===========")
    print(f"Running experiments with ε={epsilon}, δ={delta}, γ={gamma}, sensitivity={sensitivity}")
    print(f"Total rounds: {rounds}, epochs per round: {epochs_per_round}")
    print(f"Seed value for reproducibility: {seed}")
    print(f"Batch size: {batch_size}\n")


    # print("\n=== Experiment 1: No DP Noise ===")
    # start = time.time()
    # train_model_with_budget(dp_type=None, dp_params={}, total_epsilon=total_epsilon, total_delta=total_delta,
    #                         rounds=rounds, epochs_per_round=epochs_per_round)
    # print(f"Time run: {time.time() - start:.2f} seconds\n")

    # Uncomment other experiments if you want to run them after optimizing Experiment 1
    # print("\n=== Experiment 2: Gaussian DP Noise with Budget Accounting ===")
    # start = time.time()
    # train_model_with_budget(dp_type='gaussian',
    #                         dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma},
    #                         total_epsilon=total_epsilon, total_delta=total_delta,
    #                         rounds=rounds, epochs_per_round=epochs_per_round)
    # print(f"Time run: {time.time() - start:.2f} seconds\n")

    print("\n=== Experiment 3: Laplace DP Noise with Budget Accounting ===")
    start = time.time()
    train_model_with_budget(dp_type='laplace',
                            dp_params={'sensitivity': sensitivity, 'epsilon': epsilon, 'gamma': gamma},
                            total_epsilon=total_epsilon, total_delta=0.0, # Delta is typically 0 for pure Laplace
                            rounds=rounds, epochs_per_round=epochs_per_round)
    print(f"Time run: {time.time() - start:.2f} seconds\n")
    # # Count Sketch parameters
    # sketch_rows = 5
    # sketch_cols = 10000
    # csvec_blocks = 1
    
    # print(f"\n=== Experiment 4: CSVec + Gaussian DP with Budget Accounting (r={sketch_rows}, c={sketch_cols}, blocks={csvec_blocks}) ===")
    # start = time.time()
    # train_model_with_budget(dp_type='gaussian',
    #                         dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma, 'sensitivity': sensitivity}, # Pass sensitivity for CSVec context
    #                         total_epsilon=total_epsilon, total_delta=total_delta,
    #                         rounds=rounds, epochs_per_round=epochs_per_round,
    #                         use_count_sketch=True,
    #                         sketch_params={'d': sketch_rows, 'w': sketch_cols, 'numBlocks': csvec_blocks})
    # print(f"Time run: {time.time() - start:.2f} seconds\n")
    
    # print(f"\n=== Experiment 5: CSVec + Laplace DP with Budget Accounting (r={sketch_rows}, c={sketch_cols}, blocks={csvec_blocks}) ===")
    # start = time.time()
    # train_model_with_budget(dp_type='laplace',
    #                         dp_params={'delta': delta,'sensitivity': sensitivity, 'epsilon': epsilon, 'gamma': gamma},
    #                         total_epsilon=total_epsilon, total_delta=0.0, # Delta is typically 0 for pure Laplace
    #                         rounds=rounds, epochs_per_round=epochs_per_round,
    #                         use_count_sketch=True,
    #                         sketch_params={'d': sketch_rows, 'w': sketch_cols, 'numBlocks': csvec_blocks})
    # print(f"Time run: {time.time() - start:.2f} seconds\n")