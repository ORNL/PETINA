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
from PETINA import DP_Mechanisms
from PETINA.package.csvec.csvec import CSVec

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
batch_size = 240
testbatchsize = 1024
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))  # Standard MNIST normalization
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
dataset_size = len(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=testbatchsize, shuffle=False, num_workers=2, pin_memory=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getModelDimension(model):
    params = [p.detach().view(-1) for p in model.parameters()]  # Flatten each parameter
    flat_tensor = torch.cat(params)  # Concatenate into a single 1D tensor
    return len(flat_tensor)

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

# --- DP noise wrappers (These are not used for Count Sketch in this modified version directly) ---
def apply_gaussian_with_budget(grad: torch.Tensor, delta: float, epsilon: float, gamma: float) -> torch.Tensor:
    grad_np = grad.cpu().numpy() # Convert PyTorch Tensor to NumPy array
    noisy_np = DP_Mechanisms.applyDPGaussian(grad_np, delta=delta, epsilon=epsilon, gamma=gamma)
    return torch.tensor(noisy_np, dtype=torch.float32).to(device) # Convert NumPy array back to PyTorch Tensor

# --- Federated Learning Components ---
class FederatedClient:
    def __init__(self, client_id: int, train_data: torch.utils.data.Dataset, device: torch.device,
                 dp_type: str | None, dp_params: dict, use_count_sketch: bool, sketch_params: dict | None,
                 epochs_per_round: int, batch_size: int, data_per_client: int):
        self.client_id = client_id
        self.trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.device = device
        self.dp_type = dp_type
        self.dp_params = dp_params
        self.use_count_sketch = use_count_sketch
        self.sketch_params = sketch_params
        self.data_per_client = data_per_client
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

    def train_local(self, global_model_state_dict: dict) -> dict:
        """
        Performs local training on the client's data and returns either:
        - raw model update (no sketch), or
        - privatized Count Sketch table (with sketch + DP).
        """
        # 1. Save initial model parameters
        initial_model = SampleConvNet().to(self.device)
        initial_model.load_state_dict(global_model_state_dict)
        initial_flat_params = torch.cat([p.detach().view(-1) for p in initial_model.parameters()])

        # 2. Prepare accountant if needed
        accountant = GaussianAccountant() if self.dp_type == 'gaussian' else None

        # 3. Local training
        for epoch in range(self.epochs_per_round):
            self.local_model.train()
            losses = []

            for data, target in tqdm(self.trainloader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.local_model(data)
                loss = self.criterion(outputs, target)
                loss.backward()

                # 3a. Apply DP gradients (only if no Count Sketch)
                if not self.use_count_sketch and self.dp_type == 'gaussian':
                    for p in self.local_model.parameters():
                        if p.grad is not None:
                            p.grad = apply_gaussian_with_budget(
                                p.grad,
                                delta=self.dp_params.get('delta', 1e-5),
                                epsilon=self.dp_params.get('epsilon', 1.0),
                                gamma=self.dp_params.get('gamma', 1.0)
                            )

                    sample_rate = self.trainloader.batch_size / self.data_per_client
                    sigma = get_noise_multiplier(
                        target_epsilon=self.dp_params['epsilon'],
                        target_delta=self.dp_params['delta'],
                        sample_rate=sample_rate,
                        epochs=self.epochs_per_round,
                        accountant="gdp",
                    )
                    accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)

                # 3b. Clip and optimize
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                losses.append(loss.item())

            # 4. Print training progress
            loss_str = f"Train Epoch: {epoch + 1} \tLoss: {np.mean(losses):.6f}"
            if not self.use_count_sketch and self.dp_type == 'gaussian' and accountant:
                delta = self.dp_params.get('delta', 1e-5)
                epsilon_used = accountant.get_epsilon(delta=delta)
                print(f"{loss_str} (ε_accountant = {epsilon_used:.2f}, δ = {delta})")
            else:
                print(loss_str)

        # 5. Compute model update (delta)
        final_flat_params = torch.cat([p.detach().view(-1) for p in self.local_model.parameters()])
        model_update_delta = final_flat_params - initial_flat_params

        # 6. Return raw update if no Count Sketch
        if not self.use_count_sketch:
            return {
                "raw_delta": model_update_delta,
                "original_shape": final_flat_params.shape,
            }

        # 7. Apply Count Sketch with DP
        if self.dp_type and self.use_count_sketch:
            mechanism_str = self.mechanism_map.get(self.dp_type)
            if mechanism_str is None:
                raise ValueError(f"Unsupported DP noise type '{self.dp_type}' for Count Sketch DP.")

            csvec_update = DP_Mechanisms.applyCountSketch(
                domain=model_update_delta,
                num_rows=self.sketch_params['d'],
                num_cols=self.sketch_params['w'],
                epsilon=self.dp_params['epsilon'],
                delta=self.dp_params['delta'],
                mechanism=mechanism_str,
                sensitivity=self.dp_params.get('sensitivity', 1.0),
                gamma=self.dp_params.get('gamma', 0.01),
                num_blocks=self.sketch_params.get('numBlocks', 1),
                device=self.device,
                return_sketch_only=True
            )

            # Update accountant after sketching (optional)
            if self.dp_type == 'gaussian':
                sigma = get_noise_multiplier(
                    target_epsilon=self.dp_params['epsilon'],
                    target_delta=self.dp_params['delta'],
                    sample_rate=1,
                    epochs=1,
                    accountant="gdp",
                )
                accountant.step(noise_multiplier=sigma, sample_rate=1)
                epsilon = accountant.get_epsilon(delta=self.dp_params.get('delta', 1e-5))
                print(f"(ε_accountant = {epsilon:.2f}, δ = {self.dp_params['delta']})")

            return {
                "sketch_table": csvec_update.table,
                "original_shape": model_update_delta.shape,
            }


class FederatedServer:
    def __init__(self, num_clients: int, device: torch.device,
                 dp_type: str | None, dp_params: dict, use_count_sketch: bool, sketch_params: dict | None,
                 testloader: torch.utils.data.DataLoader):
        self.num_clients = num_clients
        self.global_model = SampleConvNet().to(device)
        self.model_dim = getModelDimension(self.global_model)
        self.device = device
        self.dp_type = dp_type
        self.dp_params = dp_params
        self.use_count_sketch = use_count_sketch
        self.sketch_params = sketch_params
        self.testloader = testloader
        self.clients: list[FederatedClient] = []

    def distribute_data_to_clients(self, trainset: torchvision.datasets.MNIST, batch_size: int, epochs_per_round: int):
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

    def aggregate_models(self, client_updates: list[dict]) -> dict:
        """
        Aggregates updates from clients. Handles either sketches or raw deltas.
        """
        if self.use_count_sketch:
            all_unsketched = []
            for client_update in client_updates:
                sketch_table = client_update["sketch_table"].to(self.device)
                original_shape = client_update["original_shape"]

                csvec = CSVec(
                    d=self.model_dim,
                    c=self.sketch_params['w'],
                    r=self.sketch_params['d'], 
                    numBlocks=self.sketch_params.get('numBlocks', 1),
                    device=self.device
                )
                csvec.table = sketch_table

                k = original_shape.numel() if isinstance(original_shape, torch.Size) else torch.Size(original_shape).numel()
                vec = csvec.unSketch(k=k)
                all_unsketched.append(vec)

            # Average all unsketched vectors (these are the aggregated gradients/deltas)
            avg_delta = torch.stack(all_unsketched, dim=0).mean(dim=0)

        else: # Handle raw deltas if not using count sketch (e.g., for non-DP baseline or direct DP)
            # This path expects client_updates to contain 'raw_delta' if use_count_sketch is False
            all_deltas = []
            for i, client_update in enumerate(client_updates):
                if client_update is None:
                    print(f"[Warning] Client {i} returned None. Skipping.")
                    continue
                if "raw_delta" in client_update:
                    all_deltas.append(client_update["raw_delta"].to(self.device))
                else:
                    print(f"[Warning] Client {i} update missing 'raw_delta'.")
            if not all_deltas:
                # Fallback if no raw deltas are present for non-sketch aggregation
                return self.global_model.state_dict() # Return current global model if no updates
            avg_delta = torch.stack(all_deltas, dim=0).mean(dim=0)

        # Apply the averaged delta to the global model
        aggregated_state = self.global_model.state_dict()
        idx = 0
        for name, param in aggregated_state.items():
            numel = param.numel()
            # Ensure avg_delta has enough elements and correct type
            if idx + numel > avg_delta.numel():
                print(f"Warning: avg_delta size mismatch for parameter {name}. Skipping update.")
                break
            param.data.add_(avg_delta[idx:idx + numel].view_as(param.data)) # Add delta to current global params
            idx += numel

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

            # 2. Clients train locally and send updates (sketches or raw deltas)
            client_updates = []
            for idx, client in enumerate(self.clients, start=1):
                print(f"client {idx}")
                # Pass the global model state to client_train so it can calculate delta
                update_data = client.train_local(global_model_state)
                client_updates.append(update_data)

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
    global_rounds = 5
    epochs_per_round_client = 4
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

    
    # --- Experiment 1: No DP Noise ---
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

    # # --- Experiment 2: Gaussian DP Noise with Budget Accounting ---
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
    sketch_rows = 5 
    sketch_cols = 520 #x10
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
# Total global rounds: 5, local epochs per client: 4
# Number of federated clients: 4
# Batch size: 240


# === Experiment 1: Federated Learning - No DP Noise ===
# Distributed data to 4 clients, each with 15000 samples.

# --- Global Round 1/5 ---
# client 1
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 69.62it/s]
# Train Epoch: 1  Loss: 1.184472
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.83it/s]
# Train Epoch: 2  Loss: 0.340495
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.94it/s]
# Train Epoch: 3  Loss: 0.207199
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.83it/s]
# Train Epoch: 4  Loss: 0.145336
# client 2
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.00it/s]
# Train Epoch: 1  Loss: 1.171969
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.67it/s]
# Train Epoch: 2  Loss: 0.343260
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.59it/s]
# Train Epoch: 3  Loss: 0.208252
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 83.25it/s]
# Train Epoch: 4  Loss: 0.143439
# client 3
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.30it/s]
# Train Epoch: 1  Loss: 1.158668
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.22it/s]
# Train Epoch: 2  Loss: 0.330021
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.28it/s]
# Train Epoch: 3  Loss: 0.208237
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.62it/s]
# Train Epoch: 4  Loss: 0.149074
# client 4
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 83.30it/s]
# Train Epoch: 1  Loss: 1.228096
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.56it/s]
# Train Epoch: 2  Loss: 0.337148
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.58it/s]
# Train Epoch: 3  Loss: 0.200642
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.77it/s]
# Train Epoch: 4  Loss: 0.137156
#  Global Round 1 Test Accuracy: 0.9681

# --- Global Round 2/5 ---
# client 1
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.43it/s]
# Train Epoch: 1  Loss: 0.128244
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.18it/s]
# Train Epoch: 2  Loss: 0.101199
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.70it/s]
# Train Epoch: 3  Loss: 0.071189
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.43it/s]
# Train Epoch: 4  Loss: 0.062713
# client 2
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.35it/s]
# Train Epoch: 1  Loss: 0.132237
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.96it/s]
# Train Epoch: 2  Loss: 0.098533
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.62it/s]
# Train Epoch: 3  Loss: 0.084522
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.29it/s]
# Train Epoch: 4  Loss: 0.067415
# client 3
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.59it/s]
# Train Epoch: 1  Loss: 0.132664
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.36it/s]
# Train Epoch: 2  Loss: 0.095300
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.88it/s]
# Train Epoch: 3  Loss: 0.077023
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.80it/s]
# Train Epoch: 4  Loss: 0.063543
# client 4
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.88it/s]
# Train Epoch: 1  Loss: 0.126475
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.42it/s]
# Train Epoch: 2  Loss: 0.092586
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.98it/s]
# Train Epoch: 3  Loss: 0.062731
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.50it/s]
# Train Epoch: 4  Loss: 0.051287
#  Global Round 2 Test Accuracy: 0.9840

# --- Global Round 3/5 ---
# client 1
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.62it/s]
# Train Epoch: 1  Loss: 0.066257
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.88it/s]
# Train Epoch: 2  Loss: 0.052323
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 80.29it/s]
# Train Epoch: 3  Loss: 0.048524
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.65it/s]
# Train Epoch: 4  Loss: 0.038918
# client 2
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.68it/s]
# Train Epoch: 1  Loss: 0.072451
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.46it/s]
# Train Epoch: 2  Loss: 0.060356
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 80.98it/s]
# Train Epoch: 3  Loss: 0.047347
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.91it/s]
# Train Epoch: 4  Loss: 0.040164
# client 3
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.43it/s]
# Train Epoch: 1  Loss: 0.067293
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.88it/s]
# Train Epoch: 2  Loss: 0.061286
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.20it/s]
# Train Epoch: 3  Loss: 0.046222
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.29it/s]
# Train Epoch: 4  Loss: 0.037156
# client 4
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 80.24it/s]
# Train Epoch: 1  Loss: 0.059038
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.97it/s]
# Train Epoch: 2  Loss: 0.044724
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.32it/s]
# Train Epoch: 3  Loss: 0.039689
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.82it/s]
# Train Epoch: 4  Loss: 0.031546
#  Global Round 3 Test Accuracy: 0.9886

# --- Global Round 4/5 ---
# client 1
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.22it/s]
# Train Epoch: 1  Loss: 0.051636
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.73it/s]
# Train Epoch: 2  Loss: 0.041079
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 83.31it/s]
# Train Epoch: 3  Loss: 0.031688
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.35it/s]
# Train Epoch: 4  Loss: 0.027388
# client 2
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.27it/s]
# Train Epoch: 1  Loss: 0.051828
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 83.56it/s]
# Train Epoch: 2  Loss: 0.037600
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.79it/s]
# Train Epoch: 3  Loss: 0.037842
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.74it/s]
# Train Epoch: 4  Loss: 0.027117
# client 3
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.57it/s]
# Train Epoch: 1  Loss: 0.047680
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.81it/s]
# Train Epoch: 2  Loss: 0.039887
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.01it/s]
# Train Epoch: 3  Loss: 0.030361
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.46it/s]
# Train Epoch: 4  Loss: 0.024501
# client 4
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.17it/s]
# Train Epoch: 1  Loss: 0.039860
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.24it/s]
# Train Epoch: 2  Loss: 0.033381
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.23it/s]
# Train Epoch: 3  Loss: 0.024903
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 83.15it/s]
# Train Epoch: 4  Loss: 0.018642
#  Global Round 4 Test Accuracy: 0.9895

# --- Global Round 5/5 ---
# client 1
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.65it/s]
# Train Epoch: 1  Loss: 0.040706
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.50it/s]
# Train Epoch: 2  Loss: 0.032967
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.70it/s]
# Train Epoch: 3  Loss: 0.024895
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 83.00it/s]
# Train Epoch: 4  Loss: 0.019641
# client 2
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.18it/s]
# Train Epoch: 1  Loss: 0.040502
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.73it/s]
# Train Epoch: 2  Loss: 0.034748
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.33it/s]
# Train Epoch: 3  Loss: 0.024024
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.44it/s]
# Train Epoch: 4  Loss: 0.023971
# client 3
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.43it/s]
# Train Epoch: 1  Loss: 0.038388
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.70it/s]
# Train Epoch: 2  Loss: 0.028588
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.12it/s]
# Train Epoch: 3  Loss: 0.024724
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 82.52it/s]
# Train Epoch: 4  Loss: 0.016935
# client 4
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.57it/s]
# Train Epoch: 1  Loss: 0.035495
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 83.24it/s]
# Train Epoch: 2  Loss: 0.027677
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 81.82it/s]
# Train Epoch: 3  Loss: 0.015774
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 83.13it/s]
# Train Epoch: 4  Loss: 0.015315
#  Global Round 5 Test Accuracy: 0.9896
# Federated training completed.

# Time run: 64.31 seconds


# === Experiment 2: Federated Learning - Gaussian DP Noise with Budget Accounting ===
# Distributed data to 4 clients, each with 15000 samples.

# --- Global Round 1/5 ---
# client 1
# /autofs/nccs-svm1_envoy_od/nt9/summer_2025/improving_petina/work_Intern/PETINA/PETINA/package/Opacus_budget_accountant/accountants/gdp.py:23: UserWarning: GDP accounting is experimental and can underestimate privacy expenditure.Proceed with caution. More details: https://arxiv.org/pdf/2106.02848.pdf
#   warnings.warn(
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 48.15it/s]
# Train Epoch: 1  Loss: 2.070721 (ε_accountant = 0.47, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 48.04it/s]
# Train Epoch: 2  Loss: 0.963516 (ε_accountant = 0.68, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.65it/s]
# Train Epoch: 3  Loss: 0.419111 (ε_accountant = 0.85, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.70it/s]
# Train Epoch: 4  Loss: 0.270764 (ε_accountant = 1.00, δ = 1e-05)
# client 2
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.81it/s]
# Train Epoch: 1  Loss: 2.053080 (ε_accountant = 0.47, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.60it/s]
# Train Epoch: 2  Loss: 0.902735 (ε_accountant = 0.68, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.95it/s]
# Train Epoch: 3  Loss: 0.388161 (ε_accountant = 0.85, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.70it/s]
# Train Epoch: 4  Loss: 0.264511 (ε_accountant = 1.00, δ = 1e-05)
# client 3
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 48.06it/s]
# Train Epoch: 1  Loss: 2.071677 (ε_accountant = 0.47, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 48.29it/s]
# Train Epoch: 2  Loss: 0.929197 (ε_accountant = 0.68, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.22it/s]
# Train Epoch: 3  Loss: 0.391470 (ε_accountant = 0.85, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.70it/s]
# Train Epoch: 4  Loss: 0.270914 (ε_accountant = 1.00, δ = 1e-05)
# client 4
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.81it/s]
# Train Epoch: 1  Loss: 2.048021 (ε_accountant = 0.47, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.65it/s]
# Train Epoch: 2  Loss: 0.833691 (ε_accountant = 0.68, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.93it/s]
# Train Epoch: 3  Loss: 0.367128 (ε_accountant = 0.85, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.46it/s]
# Train Epoch: 4  Loss: 0.236610 (ε_accountant = 1.00, δ = 1e-05)
#  Global Round 1 Test Accuracy: 0.9346

# --- Global Round 2/5 ---
# client 1
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.78it/s]
# Train Epoch: 1  Loss: 0.244959 (ε_accountant = 0.47, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.09it/s]
# Train Epoch: 2  Loss: 0.197265 (ε_accountant = 0.68, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 44.52it/s]
# Train Epoch: 3  Loss: 0.170509 (ε_accountant = 0.85, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.89it/s]
# Train Epoch: 4  Loss: 0.149053 (ε_accountant = 1.00, δ = 1e-05)
# client 2
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.69it/s]
# Train Epoch: 1  Loss: 0.234305 (ε_accountant = 0.47, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.89it/s]
# Train Epoch: 2  Loss: 0.190510 (ε_accountant = 0.68, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.48it/s]
# Train Epoch: 3  Loss: 0.164718 (ε_accountant = 0.85, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.24it/s]
# Train Epoch: 4  Loss: 0.145358 (ε_accountant = 1.00, δ = 1e-05)
# client 3
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.63it/s]
# Train Epoch: 1  Loss: 0.242353 (ε_accountant = 0.47, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.24it/s]
# Train Epoch: 2  Loss: 0.195584 (ε_accountant = 0.68, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.05it/s]
# Train Epoch: 3  Loss: 0.167362 (ε_accountant = 0.85, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.78it/s]
# Train Epoch: 4  Loss: 0.149508 (ε_accountant = 1.00, δ = 1e-05)
# client 4
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.35it/s]
# Train Epoch: 1  Loss: 0.230361 (ε_accountant = 0.47, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.05it/s]
# Train Epoch: 2  Loss: 0.184265 (ε_accountant = 0.68, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.83it/s]
# Train Epoch: 3  Loss: 0.159293 (ε_accountant = 0.85, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.68it/s]
# Train Epoch: 4  Loss: 0.143172 (ε_accountant = 1.00, δ = 1e-05)
#  Global Round 2 Test Accuracy: 0.9634

# --- Global Round 3/5 ---
# client 1
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 46.95it/s]
# Train Epoch: 1  Loss: 0.137222 (ε_accountant = 0.47, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.52it/s]
# Train Epoch: 2  Loss: 0.130208 (ε_accountant = 0.68, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.81it/s]
# Train Epoch: 3  Loss: 0.118676 (ε_accountant = 0.85, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.29it/s]
# Train Epoch: 4  Loss: 0.108034 (ε_accountant = 1.00, δ = 1e-05)
# client 2
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.45it/s]
# Train Epoch: 1  Loss: 0.134468 (ε_accountant = 0.47, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.47it/s]
# Train Epoch: 2  Loss: 0.122966 (ε_accountant = 0.68, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 48.05it/s]
# Train Epoch: 3  Loss: 0.117578 (ε_accountant = 0.85, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.95it/s]
# Train Epoch: 4  Loss: 0.107179 (ε_accountant = 1.00, δ = 1e-05)
# client 3
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.19it/s]
# Train Epoch: 1  Loss: 0.134176 (ε_accountant = 0.47, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.32it/s]
# Train Epoch: 2  Loss: 0.123965 (ε_accountant = 0.68, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.39it/s]
# Train Epoch: 3  Loss: 0.113315 (ε_accountant = 0.85, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.92it/s]
# Train Epoch: 4  Loss: 0.108564 (ε_accountant = 1.00, δ = 1e-05)
# client 4
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.60it/s]
# Train Epoch: 1  Loss: 0.129664 (ε_accountant = 0.47, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.54it/s]
# Train Epoch: 2  Loss: 0.121089 (ε_accountant = 0.68, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.81it/s]
# Train Epoch: 3  Loss: 0.113000 (ε_accountant = 0.85, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.72it/s]
# Train Epoch: 4  Loss: 0.108783 (ε_accountant = 1.00, δ = 1e-05)
#  Global Round 3 Test Accuracy: 0.9729

# --- Global Round 4/5 ---
# client 1
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.13it/s]
# Train Epoch: 1  Loss: 0.105907 (ε_accountant = 0.47, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.22it/s]
# Train Epoch: 2  Loss: 0.099326 (ε_accountant = 0.68, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.57it/s]
# Train Epoch: 3  Loss: 0.096601 (ε_accountant = 0.85, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.98it/s]
# Train Epoch: 4  Loss: 0.092338 (ε_accountant = 1.00, δ = 1e-05)
# client 2
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.15it/s]
# Train Epoch: 1  Loss: 0.103571 (ε_accountant = 0.47, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.20it/s]
# Train Epoch: 2  Loss: 0.099128 (ε_accountant = 0.68, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.50it/s]
# Train Epoch: 3  Loss: 0.094114 (ε_accountant = 0.85, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.67it/s]
# Train Epoch: 4  Loss: 0.091230 (ε_accountant = 1.00, δ = 1e-05)
# client 3
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.80it/s]
# Train Epoch: 1  Loss: 0.107307 (ε_accountant = 0.47, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.53it/s]
# Train Epoch: 2  Loss: 0.097652 (ε_accountant = 0.68, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.57it/s]
# Train Epoch: 3  Loss: 0.089462 (ε_accountant = 0.85, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.69it/s]
# Train Epoch: 4  Loss: 0.095253 (ε_accountant = 1.00, δ = 1e-05)
# client 4
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.73it/s]
# Train Epoch: 1  Loss: 0.101486 (ε_accountant = 0.47, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.52it/s]
# Train Epoch: 2  Loss: 0.097759 (ε_accountant = 0.68, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.56it/s]
# Train Epoch: 3  Loss: 0.092565 (ε_accountant = 0.85, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.54it/s]
# Train Epoch: 4  Loss: 0.085757 (ε_accountant = 1.00, δ = 1e-05)
#  Global Round 4 Test Accuracy: 0.9793

# --- Global Round 5/5 ---
# client 1
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.73it/s]
# Train Epoch: 1  Loss: 0.090099 (ε_accountant = 0.47, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.93it/s]
# Train Epoch: 2  Loss: 0.088848 (ε_accountant = 0.68, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.49it/s]
# Train Epoch: 3  Loss: 0.084045 (ε_accountant = 0.85, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.99it/s]
# Train Epoch: 4  Loss: 0.079273 (ε_accountant = 1.00, δ = 1e-05)
# client 2
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.38it/s]
# Train Epoch: 1  Loss: 0.085290 (ε_accountant = 0.47, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 48.33it/s]
# Train Epoch: 2  Loss: 0.082030 (ε_accountant = 0.68, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.63it/s]
# Train Epoch: 3  Loss: 0.080594 (ε_accountant = 0.85, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.69it/s]
# Train Epoch: 4  Loss: 0.077465 (ε_accountant = 1.00, δ = 1e-05)
# client 3
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.49it/s]
# Train Epoch: 1  Loss: 0.084902 (ε_accountant = 0.47, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.22it/s]
# Train Epoch: 2  Loss: 0.081735 (ε_accountant = 0.68, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 48.39it/s]
# Train Epoch: 3  Loss: 0.079290 (ε_accountant = 0.85, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.65it/s]
# Train Epoch: 4  Loss: 0.077824 (ε_accountant = 1.00, δ = 1e-05)
# client 4
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.40it/s]
# Train Epoch: 1  Loss: 0.085388 (ε_accountant = 0.47, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 48.09it/s]
# Train Epoch: 2  Loss: 0.078706 (ε_accountant = 0.68, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.51it/s]
# Train Epoch: 3  Loss: 0.079311 (ε_accountant = 0.85, δ = 1e-05)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 47.58it/s]
# Train Epoch: 4  Loss: 0.073179 (ε_accountant = 1.00, δ = 1e-05)
#  Global Round 5 Test Accuracy: 0.9803
# Federated training completed.

# Time run: 108.77 seconds


# === Experiment 3: Federated Learning - CSVec + Gaussian DP with Budget Accounting (r=5, c=520, blocks=1) ===
# Distributed data to 4 clients, each with 15000 samples.

# --- Global Round 1/5 ---
# client 1
# /autofs/nccs-svm1_envoy_od/nt9/summer_2025/improving_petina/work_Intern/PETINA/PETINA/package/Opacus_budget_accountant/accountants/gdp.py:23: UserWarning: GDP accounting is experimental and can underestimate privacy expenditure.Proceed with caution. More details: https://arxiv.org/pdf/2106.02848.pdf
#   warnings.warn(
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 66.13it/s]
# Train Epoch: 1  Loss: 1.069541
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.60it/s]
# Train Epoch: 2  Loss: 0.340146
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.09it/s]
# Train Epoch: 3  Loss: 0.191030
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.71it/s]
# Train Epoch: 4  Loss: 0.137764
# (ε_accountant = 0.99, δ = 1e-05)
# client 2
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.53it/s]
# Train Epoch: 1  Loss: 1.076283
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.82it/s]
# Train Epoch: 2  Loss: 0.337158
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 77.09it/s]
# Train Epoch: 3  Loss: 0.196584
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.80it/s]
# Train Epoch: 4  Loss: 0.137108
# (ε_accountant = 0.99, δ = 1e-05)
# client 3
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.69it/s]
# Train Epoch: 1  Loss: 1.084979
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.88it/s]
# Train Epoch: 2  Loss: 0.328009
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 77.06it/s]
# Train Epoch: 3  Loss: 0.189807
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.96it/s]
# Train Epoch: 4  Loss: 0.136896
# (ε_accountant = 0.99, δ = 1e-05)
# client 4
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.75it/s]
# Train Epoch: 1  Loss: 1.113838
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.26it/s]
# Train Epoch: 2  Loss: 0.319429
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 77.65it/s]
# Train Epoch: 3  Loss: 0.187418
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 77.23it/s]
# Train Epoch: 4  Loss: 0.130845
# (ε_accountant = 0.99, δ = 1e-05)
#  Global Round 1 Test Accuracy: 0.3754

# --- Global Round 2/5 ---
# client 1
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 77.16it/s]
# Train Epoch: 1  Loss: 0.521490
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 77.27it/s]
# Train Epoch: 2  Loss: 0.165686
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 77.17it/s]
# Train Epoch: 3  Loss: 0.116826
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 77.03it/s]
# Train Epoch: 4  Loss: 0.083478
# (ε_accountant = 0.99, δ = 1e-05)
# client 2
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 77.12it/s]
# Train Epoch: 1  Loss: 0.505551
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 77.03it/s]
# Train Epoch: 2  Loss: 0.163815
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.35it/s]
# Train Epoch: 3  Loss: 0.116649
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.38it/s]
# Train Epoch: 4  Loss: 0.089094
# (ε_accountant = 0.99, δ = 1e-05)
# client 3
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.41it/s]
# Train Epoch: 1  Loss: 0.522034
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.24it/s]
# Train Epoch: 2  Loss: 0.168678
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.87it/s]
# Train Epoch: 3  Loss: 0.118917
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.39it/s]
# Train Epoch: 4  Loss: 0.087999
# (ε_accountant = 0.99, δ = 1e-05)
# client 4
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.56it/s]
# Train Epoch: 1  Loss: 0.483646
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.23it/s]
# Train Epoch: 2  Loss: 0.162326
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.64it/s]
# Train Epoch: 3  Loss: 0.110353
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.99it/s]
# Train Epoch: 4  Loss: 0.091039
# (ε_accountant = 0.99, δ = 1e-05)
#  Global Round 2 Test Accuracy: 0.8054

# --- Global Round 3/5 ---
# client 1
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.98it/s]
# Train Epoch: 1  Loss: 0.181197
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.83it/s]
# Train Epoch: 2  Loss: 0.099513
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.59it/s]
# Train Epoch: 3  Loss: 0.077957
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.85it/s]
# Train Epoch: 4  Loss: 0.060758
# (ε_accountant = 0.99, δ = 1e-05)
# client 2
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 77.50it/s]
# Train Epoch: 1  Loss: 0.184882
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.83it/s]
# Train Epoch: 2  Loss: 0.095966
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.81it/s]
# Train Epoch: 3  Loss: 0.074547
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.69it/s]
# Train Epoch: 4  Loss: 0.059251
# (ε_accountant = 0.99, δ = 1e-05)
# client 3
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.71it/s]
# Train Epoch: 1  Loss: 0.183575
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 77.17it/s]
# Train Epoch: 2  Loss: 0.110565
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 77.55it/s]
# Train Epoch: 3  Loss: 0.079546
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.59it/s]
# Train Epoch: 4  Loss: 0.056250
# (ε_accountant = 0.99, δ = 1e-05)
# client 4
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.59it/s]
# Train Epoch: 1  Loss: 0.182139
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.89it/s]
# Train Epoch: 2  Loss: 0.098031
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.44it/s]
# Train Epoch: 3  Loss: 0.079396
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.81it/s]
# Train Epoch: 4  Loss: 0.060400
# (ε_accountant = 0.99, δ = 1e-05)
#  Global Round 3 Test Accuracy: 0.8358

# --- Global Round 4/5 ---
# client 1
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.55it/s]
# Train Epoch: 1  Loss: 0.141905
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.11it/s]
# Train Epoch: 2  Loss: 0.086269
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.94it/s]
# Train Epoch: 3  Loss: 0.063753
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 77.11it/s]
# Train Epoch: 4  Loss: 0.049364
# (ε_accountant = 0.99, δ = 1e-05)
# client 2
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.97it/s]
# Train Epoch: 1  Loss: 0.145399
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 77.55it/s]
# Train Epoch: 2  Loss: 0.087060
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.65it/s]
# Train Epoch: 3  Loss: 0.065551
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 77.29it/s]
# Train Epoch: 4  Loss: 0.051715
# (ε_accountant = 0.99, δ = 1e-05)
# client 3
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.97it/s]
# Train Epoch: 1  Loss: 0.143021
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.18it/s]
# Train Epoch: 2  Loss: 0.085552
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.15it/s]
# Train Epoch: 3  Loss: 0.071199
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.26it/s]
# Train Epoch: 4  Loss: 0.051903
# (ε_accountant = 0.99, δ = 1e-05)
# client 4
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.63it/s]
# Train Epoch: 1  Loss: 0.141740
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.61it/s]
# Train Epoch: 2  Loss: 0.088568
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.09it/s]
# Train Epoch: 3  Loss: 0.063946
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 73.65it/s]
# Train Epoch: 4  Loss: 0.047970
# (ε_accountant = 0.99, δ = 1e-05)
#  Global Round 4 Test Accuracy: 0.9398

# --- Global Round 5/5 ---
# client 1
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.93it/s]
# Train Epoch: 1  Loss: 0.111733
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.00it/s]
# Train Epoch: 2  Loss: 0.065066
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.34it/s]
# Train Epoch: 3  Loss: 0.056077
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.23it/s]
# Train Epoch: 4  Loss: 0.043168
# (ε_accountant = 0.99, δ = 1e-05)
# client 2
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.03it/s]
# Train Epoch: 1  Loss: 0.114179
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.94it/s]
# Train Epoch: 2  Loss: 0.074529
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.48it/s]
# Train Epoch: 3  Loss: 0.057278
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.51it/s]
# Train Epoch: 4  Loss: 0.043420
# (ε_accountant = 0.99, δ = 1e-05)
# client 3
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.85it/s]
# Train Epoch: 1  Loss: 0.112227
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.26it/s]
# Train Epoch: 2  Loss: 0.080156
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.04it/s]
# Train Epoch: 3  Loss: 0.051707
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.48it/s]
# Train Epoch: 4  Loss: 0.045507
# (ε_accountant = 0.99, δ = 1e-05)
# client 4
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.91it/s]
# Train Epoch: 1  Loss: 0.108944
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.71it/s]
# Train Epoch: 2  Loss: 0.072977
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.10it/s]
# Train Epoch: 3  Loss: 0.058164
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 76.21it/s]
# Train Epoch: 4  Loss: 0.042255
# (ε_accountant = 0.99, δ = 1e-05)
#  Global Round 5 Test Accuracy: 0.9356
# Federated training completed.

# Time run: 69.43 seconds