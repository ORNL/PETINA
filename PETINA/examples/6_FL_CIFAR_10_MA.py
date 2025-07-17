import math
import random
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import warnings
from numbers import Real, Integral
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

# File: PETINA/PETINA/examples/4_ML_CIFAR_10_No_MA.py
# ======================================================
#         CIFAR-10 Training with Differential Privacy
# ======================================================

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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

batch_size = 1024
testloader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

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
# These functions now correctly handle the conversion to/from NumPy for DP_Mechanisms
def apply_laplace_with_budget(grad: torch.Tensor, sensitivity: float, epsilon: float, gamma: float, accountant: BudgetAccountant) -> torch.Tensor:
    grad_np = grad.cpu().numpy() # Convert PyTorch Tensor to NumPy array
    noisy_np = DP_Mechanisms.applyDPLaplace(grad_np, sensitivity=sensitivity, epsilon=epsilon, gamma=gamma, accountant=accountant)
    return torch.tensor(noisy_np, dtype=torch.float32).to(device) # Convert NumPy array back to PyTorch Tensor

def apply_gaussian_with_budget(grad: torch.Tensor, delta: float, epsilon: float, gamma: float, accountant: BudgetAccountant) -> torch.Tensor:
    grad_np = grad.cpu().numpy() # Convert PyTorch Tensor to NumPy array
    noisy_np = DP_Mechanisms.applyDPGaussian(grad_np, delta=delta, epsilon=epsilon, gamma=gamma, accountant=accountant)
    return torch.tensor(noisy_np, dtype=torch.float32).to(device) # Convert NumPy array back to PyTorch Tensor

# --- Federated Learning Components ---

class FederatedClient:
    def __init__(self, client_id: int, train_data: torch.utils.data.Dataset, device: torch.device,
                 dp_type: str | None, dp_params: dict, use_count_sketch: bool, sketch_params: dict | None,
                 accountant: BudgetAccountant, epochs_per_round: int, batch_size: int):
        self.client_id = client_id
        self.trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.device = device
        self.dp_type = dp_type
        self.dp_params = dp_params
        self.use_count_sketch = use_count_sketch
        self.sketch_params = sketch_params
        self.accountant = accountant
        self.epochs_per_round = epochs_per_round
        self.local_model = SimpleCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=0.01, momentum=0.9)
        self.scaler = torch.amp.GradScaler('cuda' if self.device.type == 'cuda' else 'cpu')
        self.mechanism_map = {
            'gaussian': "gaussian",
            'laplace': "laplace"
        }

    def set_global_model(self, global_model_state_dict: dict):
        """Sets the client's local model to the state of the global model."""
        self.local_model.load_state_dict(global_model_state_dict)

    def get_model_parameters(self) -> dict:
        """Returns the current state dictionary of the local model."""
        return self.local_model.state_dict()

    def train_local(self) -> dict:
        """
        Performs local training on the client's data and returns the privatized
        model updates (or parameters).
        """
        self.local_model.train()
        for e in range(self.epochs_per_round):
            for inputs, targets in self.trainloader:
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.device.type):
                    outputs = self.local_model(inputs)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()

                if self.dp_type is not None:
                    self.scaler.unscale_(self.optimizer) 
                    
                    if self.use_count_sketch:
                        grad_list = [p.grad.view(-1) for p in self.local_model.parameters() if p.grad is not None]
                        if not grad_list: continue
                        flat_grad = torch.cat(grad_list)

                        mechanism_str = self.mechanism_map.get(self.dp_type)
                        if mechanism_str is None:
                            raise ValueError(f"Unsupported DP noise type '{self.dp_type}' for Count Sketch DP.")
                        
                        privatized_grad_tensor = DP_Mechanisms.applyCountSketch(
                            domain=flat_grad,
                            num_rows=self.sketch_params['d'],
                            num_cols=self.sketch_params['w'],
                            epsilon=self.dp_params['epsilon'],
                            delta=self.dp_params['delta'],
                            mechanism=mechanism_str,
                            sensitivity=self.dp_params.get('sensitivity', 1.0),
                            gamma=self.dp_params.get('gamma', 0.01),
                            num_blocks=self.sketch_params.get('numBlocks', 1),
                            device=self.device,
                            accountant=self.accountant # Pass the shared accountant object
                        )
                        
                        idx = 0
                        for p in self.local_model.parameters():
                            if p.grad is not None:
                                numel = p.grad.numel()
                                grad_slice = privatized_grad_tensor[idx:idx + numel]
                                p.grad = grad_slice.detach().clone().view_as(p.grad).to(self.device)
                                idx += numel
                    else: # Direct DP application (without Count Sketch)
                        for p in self.local_model.parameters():
                            if p.grad is None: continue
                            if self.dp_type == 'laplace':
                                p.grad = apply_laplace_with_budget(
                                    p.grad,
                                    sensitivity=self.dp_params.get('sensitivity', 1.0),
                                    epsilon=self.dp_params.get('epsilon', 1.0),
                                    gamma=self.dp_params.get('gamma', 1.0),
                                    accountant=self.accountant
                                )
                            elif self.dp_type == 'gaussian':
                                p.grad = apply_gaussian_with_budget(
                                    p.grad,
                                    delta=self.dp_params.get('delta', 1e-5),
                                    epsilon=self.dp_params.get('epsilon', 1.0),
                                    gamma=self.dp_params.get('gamma', 1.0),
                                    accountant=self.accountant
                                )
                            else:
                                raise ValueError(f"Unknown dp_type: {self.dp_type}")

                self.scaler.step(self.optimizer)
                self.scaler.update()
        
        # Return the updated local model parameters
        return self.local_model.state_dict()


class FederatedServer:
    def __init__(self, num_clients: int, total_epsilon: float, total_delta: float, device: torch.device,
                 dp_type: str | None, dp_params: dict, use_count_sketch: bool, sketch_params: dict | None,
                 testloader: torch.utils.data.DataLoader):
        self.num_clients = num_clients
        self.global_model = SimpleCNN().to(device)
        self.accountant = BudgetAccountant(epsilon=total_epsilon, delta=total_delta)
        self.device = device
        self.dp_type = dp_type
        self.dp_params = dp_params
        self.use_count_sketch = use_count_sketch
        self.sketch_params = sketch_params
        self.testloader = testloader
        self.clients: list[FederatedClient] = []

        print(f"Initialized FederatedServer with BudgetAccountant: ε={total_epsilon}, δ={total_delta}")

    def distribute_data_to_clients(self, trainset: torchvision.datasets.CIFAR10, batch_size: int, epochs_per_round: int):
        """Distributes the training data among clients and initializes client objects."""
        data_per_client = len(trainset) // self.num_clients
        
        # Create a list of Subset objects for each client
        client_data_indices = list(range(len(trainset)))
        random.shuffle(client_data_indices) # Shuffle to ensure random distribution

        for i in range(self.num_clients):
            start_idx = i * data_per_client
            end_idx = start_idx + data_per_client
            subset_indices = client_data_indices[start_idx:end_idx]
            client_subset = torch.utils.data.Subset(trainset, subset_indices)
            
            client = FederatedClient(
                client_id=i,
                train_data=client_subset,
                device=self.device,
                dp_type=self.dp_type,
                dp_params=self.dp_params,
                use_count_sketch=self.use_count_sketch,
                sketch_params=self.sketch_params,
                accountant=self.accountant, # Pass the shared accountant
                epochs_per_round=epochs_per_round,
                batch_size=batch_size
            )
            self.clients.append(client)
        print(f"Distributed data to {self.num_clients} clients, each with {data_per_client} samples.")


    def aggregate_models(self, client_model_states: list[dict]) -> dict:
        """
        Aggregates model parameters from clients using Federated Averaging.
        Assumes all clients have the same model architecture.
        """
        if not client_model_states:
            return self.global_model.state_dict()

        # Initialize aggregated state with the first client's model state
        aggregated_state = client_model_states[0].copy()

        # Sum up parameters from all other clients
        for i in range(1, len(client_model_states)):
            for key in aggregated_state:
                aggregated_state[key] += client_model_states[i][key]

        # Average the parameters
        for key in aggregated_state:
            aggregated_state[key] /= len(client_model_states)
            
        return aggregated_state

    def train_federated(self, global_rounds: int):
        """
        Orchestrates the federated learning training process.
        """
        try:
            for round_num in range(global_rounds):
                print(f"\n--- Global Round {round_num + 1}/{global_rounds} ---")

                # 1. Server sends global model to clients
                global_model_state = self.global_model.state_dict()
                for client in self.clients:
                    client.set_global_model(global_model_state)

                # 2. Clients train locally and send updates
                client_updates = []
                for client in self.clients:
                    try:
                        updated_local_model_state = client.train_local()
                        client_updates.append(updated_local_model_state)
                    except BudgetError as be:
                        print(f"Client {client.client_id} stopped due to BudgetError: {be}. Skipping this client for this round.")
                        # Optionally, you could handle this more gracefully, e.g., by not including this client's update
                        # or by stopping the entire training if a critical client runs out of budget.
                        continue
                
                if not client_updates:
                    print("No clients returned updates this round. Stopping federated training.")
                    break

                # 3. Server aggregates updates
                aggregated_state = self.aggregate_models(client_updates)

                # 4. Server updates global model
                self.global_model.load_state_dict(aggregated_state)

                # 5. Evaluate global model
                acc = evaluate(self.global_model, self.testloader)
                eps_used, delta_used = self.accountant.total()
                eps_rem, delta_rem = self.accountant.remaining()

                print(f" Global Round {round_num + 1} Test Accuracy: {acc:.4f}")
                print(f"   Total Used ε: {eps_used:.4f}, δ: {delta_used:.6f}")
                print(f"   Total Remaining ε: {eps_rem:.4f}, δ: {delta_rem:.6f}")

                if eps_rem <= 0 and delta_rem <= 0:
                    print("\nGlobal privacy budget exhausted! Stopping federated training early.")
                    break

        except BudgetError as be:
            print(f"\nFederated training stopped due to BudgetError: {be}")
        except Exception as ex:
            print(f"\nAn unexpected error occurred during federated training: {ex}")

        print("Federated training completed.\n")
        return self.global_model


if __name__ == "__main__":
    total_epsilon = 11000
    total_delta = 1-1e-9 
    global_rounds = 2 # Number of communication rounds between server and clients
    epochs_per_round_client = 3 # Number of local epochs each client runs per global round
    num_federated_clients = 4

    delta = 1e-5
    epsilon = 1.1011632828830176
    gamma = 0.01
    sensitivity = 1.0

    print("===========Parameters for Federated DP Training===========")
    print(f"Running experiments with ε={epsilon}, δ={delta}, γ={gamma}, sensitivity={sensitivity}")
    print(f"Total global rounds: {global_rounds}, local epochs per client: {epochs_per_round_client}")
    print(f"Number of federated clients: {num_federated_clients}")
    print(f"Seed value for reproducibility: {seed}")
    print(f"Batch size: {batch_size}\n")

    # --- Experiment 1: No DP Noise ---
    # print("\n=== Experiment 1: Federated Learning - No DP Noise ===")
    # server_no_dp = FederatedServer(
    #     num_clients=num_federated_clients,
    #     total_epsilon=float('inf'), # Infinite budget for no DP
    #     total_delta=1.0, # Delta is typically 1.0 for no DP
    #     device=device,
    #     dp_type=None,
    #     dp_params={},
    #     use_count_sketch=False,
    #     sketch_params=None,
    #     testloader=testloader
    # )
    # server_no_dp.distribute_data_to_clients(trainset, batch_size, epochs_per_round_client)
    # trained_global_model_no_dp = server_no_dp.train_federated(global_rounds=global_rounds)

    # # --- Experiment 2: Gaussian DP Noise with Budget Accounting ---
    # print("\n=== Experiment 2: Federated Learning - Gaussian DP Noise with Budget Accounting ===")
    # server_gaussian_dp = FederatedServer(
    #     num_clients=num_federated_clients,
    #     total_epsilon=total_epsilon,
    #     total_delta=total_delta,
    #     device=device,
    #     dp_type='gaussian',
    #     dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma, 'sensitivity': sensitivity},
    #     use_count_sketch=False,
    #     sketch_params=None,
    #     testloader=testloader
    # )
    # server_gaussian_dp.distribute_data_to_clients(trainset, batch_size, epochs_per_round_client)
    # trained_global_model_gaussian_dp = server_gaussian_dp.train_federated(global_rounds=global_rounds)

    # # --- Experiment 3: Laplace DP Noise with Budget Accounting ---
    # print("\n=== Experiment 3: Federated Learning - Laplace DP Noise with Budget Accounting ===")
    # server_laplace_dp = FederatedServer(
    #     num_clients=num_federated_clients,
    #     total_epsilon=total_epsilon,
    #     total_delta=0.0, # Delta is typically 0 for pure Laplace
    #     device=device,
    #     dp_type='laplace',
    #     dp_params={'sensitivity': sensitivity, 'epsilon': epsilon, 'gamma': gamma},
    #     use_count_sketch=False,
    #     sketch_params=None,
    #     testloader=testloader
    # )
    # server_laplace_dp.distribute_data_to_clients(trainset, batch_size, epochs_per_round_client)
    # trained_global_model_laplace_dp = server_laplace_dp.train_federated(global_rounds=global_rounds)

    # --- Experiment 4: CSVec + Gaussian DP with Budget Accounting ---
    sketch_rows = 5
    sketch_cols = 10000
    csvec_blocks = 1
    # print(f"\n=== Experiment 4: Federated Learning - CSVec + Gaussian DP with Budget Accounting (r={sketch_rows}, c={sketch_cols}, blocks={csvec_blocks}) ===")
    # server_cs_gaussian = FederatedServer(
    #     num_clients=num_federated_clients,
    #     total_epsilon=total_epsilon,
    #     total_delta=total_delta,
    #     device=device,
    #     dp_type='gaussian',
    #     dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma, 'sensitivity': sensitivity},
    #     use_count_sketch=True,
    #     sketch_params={'d': sketch_rows, 'w': sketch_cols, 'numBlocks': csvec_blocks},
    #     testloader=testloader
    # )
    # server_cs_gaussian.distribute_data_to_clients(trainset, batch_size, epochs_per_round_client)
    # trained_global_model_cs_gaussian = server_cs_gaussian.train_federated(global_rounds=global_rounds)

    # --- Experiment 5: CSVec + Laplace DP with Budget Accounting ---
    print(f"\n=== Experiment 5: Federated Learning - CSVec + Laplace DP with Budget Accounting (r={sketch_rows}, c={sketch_cols}, blocks={csvec_blocks}) ===")
    server_cs_laplace = FederatedServer(
        num_clients=num_federated_clients,
        total_epsilon=total_epsilon,
        total_delta=0.0, # Delta is typically 0 for pure Laplace
        device=device,
        dp_type='laplace',
        dp_params={'delta': delta,'sensitivity': sensitivity, 'epsilon': epsilon, 'gamma': gamma},
        use_count_sketch=True,
        sketch_params={'d': sketch_rows, 'w': sketch_cols, 'numBlocks': csvec_blocks},
        testloader=testloader
    )
    server_cs_laplace.distribute_data_to_clients(trainset, batch_size, epochs_per_round_client)
    trained_global_model_cs_laplace = server_cs_laplace.train_federated(global_rounds=global_rounds)

# -------------OUTPUT-----------------
# Using device: cuda
# Device name: NVIDIA GeForce RTX 3060 Ti
# Running experiments with ε=1.1011632828830176, δ=1e-05, γ=0.01, sensitivity=1.0
# Total global rounds: 2, local epochs per client: 3
# Number of federated clients: 4
# Seed value for reproducibility: 42
# Batch size: 1024


# === Experiment 1: Federated Learning - No DP Noise ===
# Initialized FederatedServer with BudgetAccountant: ε=inf, δ=1.0
# Distributed data to 4 clients, each with 12500 samples.

# --- Global Round 1/2 ---
#  Global Round 1 Test Accuracy: 0.2441
#    Total Used ε: 0.0000, δ: 0.000000
#    Total Remaining ε: inf, δ: 1.000000

# --- Global Round 2/2 ---
#  Global Round 2 Test Accuracy: 0.3428
#    Total Used ε: 0.0000, δ: 0.000000
#    Total Remaining ε: inf, δ: 1.000000
# Federated training completed.


# === Experiment 2: Federated Learning - Gaussian DP Noise with Budget Accounting ===
# Initialized FederatedServer with BudgetAccountant: ε=11000, δ=0.999999999
# Distributed data to 4 clients, each with 12500 samples.

# --- Global Round 1/2 ---
#  Global Round 1 Test Accuracy: 0.2322
#    Total Used ε: 1374.2518, δ: 0.012403
#    Total Remaining ε: 9625.7501, δ: 1.000000

# --- Global Round 2/2 ---
#  Global Round 2 Test Accuracy: 0.3411
#    Total Used ε: 2748.5036, δ: 0.024651
#    Total Remaining ε: 8251.4949, δ: 1.000000
# Federated training completed.


# === Experiment 3: Federated Learning - Laplace DP Noise with Budget Accounting ===
# Initialized FederatedServer with BudgetAccountant: ε=11000, δ=0.0
# Distributed data to 4 clients, each with 12500 samples.

# --- Global Round 1/2 ---
#  Global Round 1 Test Accuracy: 0.2673
#    Total Used ε: 1374.2518, δ: 0.000000
#    Total Remaining ε: 9625.7501, δ: 0.000000

# --- Global Round 2/2 ---
#  Global Round 2 Test Accuracy: 0.3441
#    Total Used ε: 2748.5036, δ: 0.000000
#    Total Remaining ε: 8251.4949, δ: 0.000000
# Federated training completed.

# === Experiment 4: Federated Learning - CSVec + Gaussian DP with Budget Accounting (r=5, c=10000, blocks=1) ===
# Initialized FederatedServer with BudgetAccountant: ε=11000, δ=0.999999999
# Distributed data to 4 clients, each with 12500 samples.

# --- Global Round 1/2 ---
#  Global Round 1 Test Accuracy: 0.2387
#    Total Used ε: 171.7815, δ: 0.001559
#    Total Remaining ε: 10828.2142, δ: 1.000000

# --- Global Round 2/2 ---
#  Global Round 2 Test Accuracy: 0.3486
#    Total Used ε: 343.5629, δ: 0.003115
#    Total Remaining ε: 10656.4336, δ: 1.000000
# Federated training completed.

# === Experiment 5: Federated Learning - CSVec + Laplace DP with Budget Accounting (r=5, c=10000, blocks=1) ===
# Initialized FederatedServer with BudgetAccountant: ε=11000, δ=0.0
# Distributed data to 4 clients, each with 12500 samples.

# --- Global Round 1/2 ---
#  Global Round 1 Test Accuracy: 0.2464
#    Total Used ε: 171.7815, δ: 0.000000
#    Total Remaining ε: 10828.2142, δ: 0.000000

# --- Global Round 2/2 ---
#  Global Round 2 Test Accuracy: 0.3470
#    Total Used ε: 343.5629, δ: 0.000000
#    Total Remaining ε: 10656.4336, δ: 0.000000
# Federated training completed.
