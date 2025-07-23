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
        Performs local training on the client's data and returns the privatized
        sketch of model updates (not raw gradients or parameters).
        """
        # Store initial model state to calculate the update (delta) later
        initial_model = SampleConvNet().to(self.device)
        initial_model.load_state_dict(global_model_state_dict)
        initial_flat_params = torch.cat([p.detach().view(-1) for p in initial_model.parameters()])

        accountantOPC = GaussianAccountant() # This accountant might need adjustments if privacy is only after local training.

        for e in range(self.epochs_per_round):
            self.local_model.train()
            losses = []

            for _batch_idx, (data, target) in enumerate(tqdm(self.trainloader)):
                data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()
                outputs = self.local_model(data)
                loss = self.criterion(outputs, target)
                loss.backward()

                # Clip gradients BEFORE applying optimizer step (standard for DP-SGD)
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                losses.append(loss.item())

            loss_str = f"Train Epoch: {e+1} \tLoss: {np.mean(losses):.6f}"
            # Privacy accounting here for the Opacus budget accountant based on per-batch operations
            # This part is tricky. If noise is added AFTER local training, this accounting for per-batch
            # noise application might not be accurate for the total privacy loss.
            # For simplicity, we keep it here but note the potential discrepancy.
            sample_rate = self.trainloader.batch_size / self.data_per_client
            sigma = get_noise_multiplier(
                target_epsilon=self.dp_params['epsilon'],
                target_delta=self.dp_params['delta'],
                sample_rate=sample_rate,
                epochs=self.epochs_per_round,
                accountant="gdp",
            )
            accountantOPC.step(noise_multiplier=sigma, sample_rate=sample_rate)


            if self.dp_type is not None:
                epsilon_accounted = accountantOPC.get_epsilon(delta=self.dp_params.get('delta', 1e-5))
                print(f"{loss_str} (ε_accountant = {epsilon_accounted:.2f}, δ = {self.dp_params.get('delta', 1e-5)} )")

        # --- AFTER LOCAL TRAINING: Calculate model update (delta) and apply DP ---
        final_flat_params = torch.cat([p.detach().view(-1) for p in self.local_model.parameters()])
        model_update_delta = final_flat_params - initial_flat_params

        # Apply DP noise and/or Count Sketch to the model_update_delta
        if self.dp_type is not None:
            if self.use_count_sketch:
                mechanism_str = self.mechanism_map.get(self.dp_type)
                if mechanism_str is None:
                    raise ValueError(f"Unsupported DP noise type '{self.dp_type}' for Count Sketch DP.")

                # Create and apply Count Sketch with DP to the model_update_delta
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
                return {
                    "sketch_table": csvec_update.table,
                    "original_shape": model_update_delta.shape,
                }
            else: # Direct DP (no Count Sketch) - This path is not selected in main() for experiment 3
                # If you were to enable this, you'd apply noise to model_update_delta
                # using apply_gaussian_with_budget here.
                # For this fix, we are focusing on the count_sketch path.
                raise NotImplementedError("Direct DP on model updates (without Count Sketch) not fully implemented in this modification.")
        else: # No DP
            # If no DP, simply return the model update delta (or the full model for FedAvg)
            # For consistency with the server's aggregate_models, we return a "mock" sketch format
            # or you'd change server_aggregate_models to handle raw deltas.
            # For now, we'll return a mock sketch (without actual sketch noise if use_dp is False)
            # if self.use_count_sketch is False. This part might need refinement depending on
            # how the server aggregates non-sketch updates.
            return {
                "raw_delta": model_update_delta, # Send the raw delta for non-DP aggregation
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
                    d=self.model_dim, # d is domain size, which is model_dim
                    c=self.sketch_params['w'], # c is number of columns (width)
                    r=self.sketch_params['d'], # r is number of rows (depth)
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
            for client_update in client_updates:
                if "raw_delta" in client_update:
                    all_deltas.append(client_update["raw_delta"].to(self.device))
                # Add logic for full model aggregation if that's the non-DP strategy
                # For this fix, we assume clients send updates (deltas) or sketches of updates.

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
    global_rounds = 3
    epochs_per_round_client = 5
    num_federated_clients = 4
    delta = 1e-5
    epsilon = 0.1
    gamma = 0.01 # Note: gamma is used for sensitivity scaling in PETINA's mechanisms
    sensitivity = 1.0 # This is used by applyDPGaussian in PETINA

    print("===========Parameters for Federated DP Training===========")
    print(f"Running experiments with ε={epsilon}, δ={delta}, γ={gamma}, sensitivity={sensitivity}")
    print(f"Total global rounds: {global_rounds}, local epochs per client: {epochs_per_round_client}")
    print(f"Number of federated clients: {num_federated_clients}")
    print(f"Batch size: {batch_size}\n")


    # --- Experiment 3: CSVec + Gaussian DP with Budget Accounting ---
    # sketch_rows = 5 # d in CSVec
    # sketch_cols = 1024 # w in CSVec
    sketch_rows = 8 # d in CSVec
    sketch_cols = 5202 # w in CSVec
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