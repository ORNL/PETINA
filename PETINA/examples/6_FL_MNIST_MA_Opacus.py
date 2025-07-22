import random
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

# === PETINA Libraries ===
from PETINA.package.Opacus_budget_accountant.accountants.gdp import GaussianAccountant
from PETINA.package.Opacus_budget_accountant.accountants.utils import get_noise_multiplier
from PETINA import  DP_Mechanisms

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

# --- SampleConvNet Model ---
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
        for data, target in dataloader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    return correct / total

# --- DP noise wrappers ---
def apply_gaussian_with_budget(grad: torch.Tensor, delta: float, epsilon: float, gamma: float) -> torch.Tensor:
    grad_np = grad.cpu().numpy() # Convert PyTorch Tensor to NumPy array
    noisy_np = DP_Mechanisms.applyDPGaussian(grad_np, delta=delta, epsilon=epsilon, gamma=gamma)
    return torch.tensor(noisy_np, dtype=torch.float32).to(device) # Convert NumPy array back to PyTorch Tensor

# --- Federated Learning Components ---
class FederatedClient:
    def __init__(self, client_id: int, train_data: torch.utils.data.Dataset, device: torch.device,
                 dp_type: str | None, dp_params: dict, use_count_sketch: bool, sketch_params: dict | None,
                 epochs_per_round: int, batch_size: int,data_per_client: int):
        self.client_id = client_id
        self.trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.device = device
        self.dp_type = dp_type
        self.dp_params = dp_params
        self.use_count_sketch = use_count_sketch
        self.sketch_params = sketch_params
        self.data_per_client=data_per_client
        self.epochs_per_round = epochs_per_round
        self.local_model = SampleConvNet().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=0.25, momentum=0)
        self.mechanism_map = {
            'gaussian': "gaussian"
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
        accountantOPC = GaussianAccountant()
        for e in range(self.epochs_per_round):
            self.local_model.train()
            losses = []
            for _batch_idx, (data, target)  in enumerate(tqdm(self.trainloader)):
                data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()
                outputs = self.local_model(data)
                loss = self.criterion(outputs, target)
                loss.backward()
                if self.dp_type is not None:                   
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
                            device=self.device
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
                            if self.dp_type == 'gaussian':
                                p.grad = apply_gaussian_with_budget(
                                    p.grad,
                                    delta=self.dp_params.get('delta', 1e-5),
                                    epsilon=self.dp_params.get('epsilon', 1.0),
                                    gamma=self.dp_params.get('gamma', 1.0)
                                    # accountant=self.accountant
                                )
                            else:
                                raise ValueError(f"Unknown dp_type: {self.dp_type}")
                    sample_rate = self.trainloader.batch_size / self.data_per_client            
                    sigma = get_noise_multiplier(
                        target_epsilon=self.dp_params['epsilon'],
                        target_delta=self.dp_params['delta'],
                        sample_rate=sample_rate,
                        epochs=self.epochs_per_round,
                        accountant="gdp",
                    )
                    accountantOPC.step(noise_multiplier=sigma, sample_rate=sample_rate)
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                losses.append(loss.item())
            loss_str = f"Train Epoch: {e+1} \tLoss: {np.mean(losses):.6f}"
            if self.dp_type is not None:
                epsilon = accountantOPC.get_epsilon(delta=self.dp_params.get('delta', 1e-5))
                print(f"{loss_str} (ε_accountant = {epsilon:.2f}, δ = {self.dp_params.get('delta', 1e-5)} )")
        return self.local_model.state_dict()


class FederatedServer:
    def __init__(self, num_clients: int, device: torch.device,
                 dp_type: str | None, dp_params: dict, use_count_sketch: bool, sketch_params: dict | None,
                 testloader: torch.utils.data.DataLoader):
        self.num_clients = num_clients
        self.global_model = SampleConvNet().to(device)
        self.device = device
        self.dp_type = dp_type
        self.dp_params = dp_params
        self.use_count_sketch = use_count_sketch
        self.sketch_params = sketch_params
        self.testloader = testloader
        self.clients: list[FederatedClient] = []

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
                epochs_per_round=epochs_per_round,
                batch_size=batch_size,
                data_per_client=data_per_client
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
        for round_num in range(global_rounds):
            print(f"\n--- Global Round {round_num + 1}/{global_rounds} ---")

            # 1. Server sends global model to clients
            global_model_state = self.global_model.state_dict()
            for client in self.clients:
                client.set_global_model(global_model_state)

            # 2. Clients train locally and send updates
            client_updates = []
            for idx, client in enumerate(self.clients, start=1):
                print(f"client {idx}")
                updated_local_model_state = client.train_local()
                client_updates.append(updated_local_model_state)            
            if not client_updates:
                print("No clients returned updates this round. Stopping federated training.")
                break

            # 3. Server aggregates updates
            aggregated_state = self.aggregate_models(client_updates)

            # 4. Server updates global model
            self.global_model.load_state_dict(aggregated_state)

            # 5. Evaluate global model
            acc = evaluate(self.global_model, self.testloader)
            print(f" Global Round {round_num + 1} Test Accuracy: {acc:.4f}")
        print("Federated training completed.\n")
        return self.global_model


def main():
    global_rounds = 3 
    epochs_per_round_client = 5 
    num_federated_clients = 4
    delta = 1e-5
    epsilon = 1
    gamma = 0.01
    sensitivity = 1.0

    print("===========Parameters for Federated DP Training===========")
    print(f"Running experiments with ε={epsilon}, δ={delta}, γ={gamma}, sensitivity={sensitivity}")
    print(f"Total global rounds: {global_rounds}, local epochs per client: {epochs_per_round_client}")
    print(f"Number of federated clients: {num_federated_clients}")
    print(f"Batch size: {batch_size}\n")

    # # --- Experiment 1: No DP Noise ---
    print("\n=== Experiment 1: Federated Learning - No DP Noise ===")
    start = time.time()
    server_no_dp = FederatedServer(
        num_clients=num_federated_clients,
        device=device,
        dp_type=None,
        dp_params={},
        use_count_sketch=False,
        sketch_params=None,
        testloader=testloader
    )
    server_no_dp.distribute_data_to_clients(trainset, batch_size, epochs_per_round_client)
    trained_global_model_no_dp = server_no_dp.train_federated(global_rounds=global_rounds)
    print(f"Time run: {time.time() - start:.2f} seconds\n")

    # --- Experiment 2: Gaussian DP Noise with Budget Accounting ---
    print("\n=== Experiment 2: Federated Learning - Gaussian DP Noise with Budget Accounting ===")
    start = time.time()
    server_gaussian_dp = FederatedServer(
        num_clients=num_federated_clients,
        device=device,
        dp_type='gaussian',
        dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma, 'sensitivity': sensitivity},
        use_count_sketch=False,
        sketch_params=None,
        testloader=testloader
    )
    server_gaussian_dp.distribute_data_to_clients(trainset, batch_size, epochs_per_round_client)
    trained_global_model_gaussian_dp = server_gaussian_dp.train_federated(global_rounds=global_rounds)
    print(f"Time run: {time.time() - start:.2f} seconds\n")

    # --- Experiment 3: CSVec + Gaussian DP with Budget Accounting ---
    sketch_rows = 3
    sketch_cols = 2048
    csvec_blocks = 1
    print(f"\n=== Experiment 3: Federated Learning - CSVec + Gaussian DP with Budget Accounting (r={sketch_rows}, c={sketch_cols}, blocks={csvec_blocks}) ===")
    start = time.time()
    server_cs_gaussian = FederatedServer(
        num_clients=num_federated_clients,
        device=device,
        dp_type='gaussian',
        dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma, 'sensitivity': sensitivity},
        use_count_sketch=True,
        sketch_params={'d': sketch_rows, 'w': sketch_cols, 'numBlocks': csvec_blocks},
        testloader=testloader
    )
    server_cs_gaussian.distribute_data_to_clients(trainset, batch_size, epochs_per_round_client)
    trained_global_model_cs_gaussian = server_cs_gaussian.train_federated(global_rounds=global_rounds)
    print(f"Time run: {time.time() - start:.2f} seconds\n")

if __name__ == "__main__":
    main()

# --------------OUTPUT--------------
# ===========Parameters for Federated DP Training===========
# Running experiments with ε=1, δ=1e-05, γ=0.01, sensitivity=1.0
# Total global rounds: 3, local epochs per client: 5
# Number of federated clients: 4
# Batch size: 240


# === Experiment 1: Federated Learning - No DP Noise ===
# Distributed data to 4 clients, each with 15000 samples.

# --- Global Round 1/3 ---
# client 1
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.11it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 56.68it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 58.16it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:03<00:00, 15.77it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 55.36it/s]
# client 2
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.54it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 55.97it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 59.31it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 55.04it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 56.67it/s]
# client 3
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 55.98it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 56.05it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 56.12it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 56.51it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 55.19it/s]
# client 4
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 57.01it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 53.42it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 54.25it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 56.22it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 56.99it/s]
#  Global Round 1 Test Accuracy: 0.9742

# --- Global Round 2/3 ---
# client 1
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 54.33it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 56.71it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 56.70it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 57.22it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 48.61it/s]
# client 2
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 54.68it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 56.25it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 54.90it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 54.61it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 57.28it/s]
# client 3
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 54.64it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:04<00:00, 15.49it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 58.77it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 56.96it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 56.99it/s]
# client 4
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 52.10it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 57.87it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 57.32it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 57.58it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 56.45it/s]
#  Global Round 2 Test Accuracy: 0.9877

# --- Global Round 3/3 ---
# client 1
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 46.06it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 57.39it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 56.45it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 54.75it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 55.41it/s]
# client 2
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 53.87it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 56.82it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 54.01it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 56.11it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 58.87it/s]
# client 3
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 57.75it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 53.01it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 57.16it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 57.74it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 56.05it/s]
# client 4
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 56.28it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 57.72it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 56.62it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 50.83it/s]
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:04<00:00, 15.37it/s]
#  Global Round 3 Test Accuracy: 0.9863
# Federated training completed.

# Time run: 79.49 seconds


# === Experiment 2: Federated Learning - Gaussian DP Noise with Budget Accounting ===
# Distributed data to 4 clients, each with 15000 samples.

# --- Global Round 1/3 ---
# client 1
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 37.90it/s]
# Train Epoch: 1  Loss: 2.008604 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 36.30it/s]
# Train Epoch: 2  Loss: 0.800889 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 38.91it/s]
# Train Epoch: 3  Loss: 0.374333 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 36.04it/s]
# Train Epoch: 4  Loss: 0.266706 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 36.28it/s]
# Train Epoch: 5  Loss: 0.213733 (ε_accountant = 1.00, δ = 1e-05 )
# client 2
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 37.69it/s]
# Train Epoch: 1  Loss: 2.026376 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 33.09it/s]
# Train Epoch: 2  Loss: 0.839365 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 40.19it/s]
# Train Epoch: 3  Loss: 0.408879 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 37.63it/s]
# Train Epoch: 4  Loss: 0.292216 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.58it/s]
# Train Epoch: 5  Loss: 0.216831 (ε_accountant = 1.00, δ = 1e-05 )
# client 3
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 40.50it/s]
# Train Epoch: 1  Loss: 2.036679 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 36.16it/s]
# Train Epoch: 2  Loss: 0.851812 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 38.45it/s]
# Train Epoch: 3  Loss: 0.384128 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 37.10it/s]
# Train Epoch: 4  Loss: 0.258313 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 40.29it/s]
# Train Epoch: 5  Loss: 0.205575 (ε_accountant = 1.00, δ = 1e-05 )
# client 4
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 38.27it/s]
# Train Epoch: 1  Loss: 2.001991 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 36.66it/s]
# Train Epoch: 2  Loss: 0.812047 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 38.34it/s]
# Train Epoch: 3  Loss: 0.393346 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:04<00:00, 13.59it/s]
# Train Epoch: 4  Loss: 0.262268 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.09it/s]
# Train Epoch: 5  Loss: 0.197919 (ε_accountant = 1.00, δ = 1e-05 )
#  Global Round 1 Test Accuracy: 0.9477

# --- Global Round 2/3 ---
# client 1
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 38.11it/s]
# Train Epoch: 1  Loss: 0.207277 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 39.14it/s]
# Train Epoch: 2  Loss: 0.166823 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 36.62it/s]
# Train Epoch: 3  Loss: 0.154131 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 37.28it/s]
# Train Epoch: 4  Loss: 0.132329 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 35.39it/s]
# Train Epoch: 5  Loss: 0.117351 (ε_accountant = 1.00, δ = 1e-05 )
# client 2
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 38.97it/s]
# Train Epoch: 1  Loss: 0.209526 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 36.86it/s]
# Train Epoch: 2  Loss: 0.169170 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 35.26it/s]
# Train Epoch: 3  Loss: 0.150924 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 41.18it/s]
# Train Epoch: 4  Loss: 0.134954 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 36.85it/s]
# Train Epoch: 5  Loss: 0.127365 (ε_accountant = 1.00, δ = 1e-05 )
# client 3
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 37.03it/s]
# Train Epoch: 1  Loss: 0.201133 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 37.93it/s]
# Train Epoch: 2  Loss: 0.179501 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 36.91it/s]
# Train Epoch: 3  Loss: 0.153214 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 37.35it/s]
# Train Epoch: 4  Loss: 0.129064 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 36.01it/s]
# Train Epoch: 5  Loss: 0.121752 (ε_accountant = 1.00, δ = 1e-05 )
# client 4
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 37.12it/s]
# Train Epoch: 1  Loss: 0.195903 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:04<00:00, 13.60it/s]
# Train Epoch: 2  Loss: 0.168832 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 36.71it/s]
# Train Epoch: 3  Loss: 0.151274 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 38.26it/s]
# Train Epoch: 4  Loss: 0.128678 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 38.01it/s]
# Train Epoch: 5  Loss: 0.122718 (ε_accountant = 1.00, δ = 1e-05 )
#  Global Round 2 Test Accuracy: 0.9683

# --- Global Round 3/3 ---
# client 1
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 40.28it/s]
# Train Epoch: 1  Loss: 0.118371 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 37.96it/s]
# Train Epoch: 2  Loss: 0.114724 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 36.70it/s]
# Train Epoch: 3  Loss: 0.105148 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 36.03it/s]
# Train Epoch: 4  Loss: 0.097353 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 38.03it/s]
# Train Epoch: 5  Loss: 0.094452 (ε_accountant = 1.00, δ = 1e-05 )
# client 2
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 37.11it/s]
# Train Epoch: 1  Loss: 0.116116 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.41it/s]
# Train Epoch: 2  Loss: 0.106265 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 39.69it/s]
# Train Epoch: 3  Loss: 0.102122 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 37.72it/s]
# Train Epoch: 4  Loss: 0.099782 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 32.58it/s]
# Train Epoch: 5  Loss: 0.104922 (ε_accountant = 1.00, δ = 1e-05 )
# client 3
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 31.79it/s]
# Train Epoch: 1  Loss: 0.111779 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 37.66it/s]
# Train Epoch: 2  Loss: 0.110022 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 33.33it/s]
# Train Epoch: 3  Loss: 0.097775 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 37.82it/s]
# Train Epoch: 4  Loss: 0.092728 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:04<00:00, 13.57it/s]
# Train Epoch: 5  Loss: 0.090417 (ε_accountant = 1.00, δ = 1e-05 )
# client 4
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 35.10it/s]
# Train Epoch: 1  Loss: 0.111512 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 37.99it/s]
# Train Epoch: 2  Loss: 0.106214 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 33.75it/s]
# Train Epoch: 3  Loss: 0.101892 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 39.60it/s]
# Train Epoch: 4  Loss: 0.094356 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 35.46it/s]
# Train Epoch: 5  Loss: 0.084867 (ε_accountant = 1.00, δ = 1e-05 )
#  Global Round 3 Test Accuracy: 0.9783
# Federated training completed.

# Time run: 113.73 seconds


# === Experiment 3: Federated Learning - CSVec + Gaussian DP with Budget Accounting (r=3, c=2048, blocks=1) ===      
# Distributed data to 4 clients, each with 15000 samples.

# --- Global Round 1/3 ---
# client 1
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:02<00:00, 29.72it/s]
# Train Epoch: 1  Loss: 1.862233 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.04it/s]
# Train Epoch: 2  Loss: 0.841057 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 32.53it/s]
# Train Epoch: 3  Loss: 0.458523 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 36.66it/s]
# Train Epoch: 4  Loss: 0.285993 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 35.41it/s]
# Train Epoch: 5  Loss: 0.215053 (ε_accountant = 1.00, δ = 1e-05 )
# client 2
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:02<00:00, 30.89it/s]
# Train Epoch: 1  Loss: 1.840052 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 36.57it/s]
# Train Epoch: 2  Loss: 0.770293 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 35.02it/s]
# Train Epoch: 3  Loss: 0.412076 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:02<00:00, 31.05it/s]
# Train Epoch: 4  Loss: 0.265304 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 35.62it/s]
# Train Epoch: 5  Loss: 0.201000 (ε_accountant = 1.00, δ = 1e-05 )
# client 3
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.25it/s]
# Train Epoch: 1  Loss: 1.877160 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:05<00:00, 12.50it/s]
# Train Epoch: 2  Loss: 0.829515 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 33.58it/s]
# Train Epoch: 3  Loss: 0.456321 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:02<00:00, 31.20it/s]
# Train Epoch: 4  Loss: 0.298219 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:02<00:00, 29.85it/s]
# Train Epoch: 5  Loss: 0.235440 (ε_accountant = 1.00, δ = 1e-05 )
# client 4
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.80it/s]
# Train Epoch: 1  Loss: 1.867801 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 33.16it/s]
# Train Epoch: 2  Loss: 0.856333 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 32.74it/s]
# Train Epoch: 3  Loss: 0.489204 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 36.56it/s]
# Train Epoch: 4  Loss: 0.316396 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.06it/s]
# Train Epoch: 5  Loss: 0.241106 (ε_accountant = 1.00, δ = 1e-05 )
#  Global Round 1 Test Accuracy: 0.9470

# --- Global Round 2/3 ---
# client 1
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 33.30it/s]
# Train Epoch: 1  Loss: 0.196635 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.50it/s]
# Train Epoch: 2  Loss: 0.169322 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.30it/s]
# Train Epoch: 3  Loss: 0.142538 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 33.79it/s]
# Train Epoch: 4  Loss: 0.132249 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.68it/s]
# Train Epoch: 5  Loss: 0.119608 (ε_accountant = 1.00, δ = 1e-05 )
# client 2
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 35.50it/s]
# Train Epoch: 1  Loss: 0.187423 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 35.42it/s]
# Train Epoch: 2  Loss: 0.158227 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 32.72it/s]
# Train Epoch: 3  Loss: 0.139465 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:04<00:00, 13.33it/s]
# Train Epoch: 4  Loss: 0.121344 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 36.02it/s]
# Train Epoch: 5  Loss: 0.109474 (ε_accountant = 1.00, δ = 1e-05 )
# client 3
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 35.03it/s]
# Train Epoch: 1  Loss: 0.202980 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 32.47it/s]
# Train Epoch: 2  Loss: 0.170586 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.11it/s]
# Train Epoch: 3  Loss: 0.149906 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 35.51it/s]
# Train Epoch: 4  Loss: 0.133124 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.96it/s]
# Train Epoch: 5  Loss: 0.120912 (ε_accountant = 1.00, δ = 1e-05 )
# client 4
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 35.31it/s]
# Train Epoch: 1  Loss: 0.202284 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 35.23it/s]
# Train Epoch: 2  Loss: 0.162440 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.52it/s]
# Train Epoch: 3  Loss: 0.148657 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 33.99it/s]
# Train Epoch: 4  Loss: 0.130139 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 35.18it/s]
# Train Epoch: 5  Loss: 0.115169 (ε_accountant = 1.00, δ = 1e-05 )
#  Global Round 2 Test Accuracy: 0.9705

# --- Global Round 3/3 ---
# client 1
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 36.05it/s]
# Train Epoch: 1  Loss: 0.114338 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 35.15it/s]
# Train Epoch: 2  Loss: 0.108200 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.31it/s]
# Train Epoch: 3  Loss: 0.097708 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 32.73it/s]
# Train Epoch: 4  Loss: 0.092722 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 33.99it/s]
# Train Epoch: 5  Loss: 0.087374 (ε_accountant = 1.00, δ = 1e-05 )
# client 2
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:04<00:00, 13.14it/s]
# Train Epoch: 1  Loss: 0.109125 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 36.26it/s]
# Train Epoch: 2  Loss: 0.101012 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 33.97it/s]
# Train Epoch: 3  Loss: 0.092691 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.45it/s]
# Train Epoch: 4  Loss: 0.089821 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.79it/s]
# Train Epoch: 5  Loss: 0.081986 (ε_accountant = 1.00, δ = 1e-05 )
# client 3
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.86it/s]
# Train Epoch: 1  Loss: 0.115574 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.28it/s]
# Train Epoch: 2  Loss: 0.107436 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.62it/s]
# Train Epoch: 3  Loss: 0.101259 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.62it/s]
# Train Epoch: 4  Loss: 0.097259 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 32.14it/s]
# Train Epoch: 5  Loss: 0.091322 (ε_accountant = 1.00, δ = 1e-05 )
# client 4
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 33.53it/s]
# Train Epoch: 1  Loss: 0.114353 (ε_accountant = 0.42, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 33.14it/s]
# Train Epoch: 2  Loss: 0.102778 (ε_accountant = 0.61, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 35.38it/s]
# Train Epoch: 3  Loss: 0.100024 (ε_accountant = 0.76, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 34.26it/s]
# Train Epoch: 4  Loss: 0.093464 (ε_accountant = 0.89, δ = 1e-05 )
# 100%|██████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 35.12it/s]
# Train Epoch: 5  Loss: 0.087066 (ε_accountant = 1.00, δ = 1e-05 )
#  Global Round 3 Test Accuracy: 0.9788
# Federated training completed.

# Time run: 122.32 seconds