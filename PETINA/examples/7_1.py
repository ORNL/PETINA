import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import copy
from multiprocessing import Process, Queue
from PETINA import DP_Mechanisms, BudgetAccountant, BudgetError
import multiprocessing as mp



# ============ GPU Assignment for Each Client ============
# Assumes you have 3 GPUs available (cuda:0, cuda:1, cuda:2)
# Modify based on your actual system

device_list = ["cuda:0", "cuda:1", "cuda:2"]

# ============ Data and Preprocessing ============
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

num_clients = 3
client_data_size = len(trainset) // num_clients
split_sizes = [client_data_size] * (num_clients - 1) + [len(trainset) - client_data_size * (num_clients - 1)]
client_datasets = random_split(trainset, split_sizes)

def get_loader(dataset):
    return DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

# ============ Model ============
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 10)
        )

    def forward(self, x):
        return self.net(x)

# ============ Evaluation ============
def evaluate(model, testloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ============ Client Training ============
def client_worker(model_state_dict, dataset, device, dp_type, dp_params, accountant, epochs, queue, client_id):
    torch.cuda.set_device(device)
    model = TinyCNN().to(device)
    model.load_state_dict(model_state_dict)
    model.train()
    loader = get_loader(dataset)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            for p in model.parameters():
                if p.grad is None:
                    continue
                grad = p.grad.cpu().numpy()
                if dp_type == 'laplace':
                    noisy_grad = DP_Mechanisms.applyDPLaplace(
                        grad, sensitivity=dp_params.get('sensitivity', 1.0),
                        epsilon=dp_params['epsilon'], gamma=dp_params['gamma'], accountant=accountant)
                elif dp_type == 'gaussian':
                    noisy_grad = DP_Mechanisms.applyDPGaussian(
                        grad, delta=dp_params['delta'], epsilon=dp_params['epsilon'],
                        gamma=dp_params['gamma'], accountant=accountant)
                else:
                    noisy_grad = grad
                p.grad = torch.tensor(noisy_grad, dtype=p.grad.dtype).to(device)
            optimizer.step()

    queue.put((client_id, model.state_dict()))

# ============ Federated Averaging ============
def federated_average(states):
    avg_state = copy.deepcopy(states[0])
    for key in avg_state:
        avg_state[key] = torch.stack([s[key].float() for s in states], 0).mean(0)
    return avg_state

# ============ Federated Training Loop ============
def federated_training(rounds=3, epochs=1):
    global_model = TinyCNN().to("cuda:0")
    global_state = global_model.state_dict()

    accountant = BudgetAccountant(epsilon=5.0, delta=1)
    dp_type = "gaussian"
    dp_params = {"delta": 1e-6, "epsilon": 0.01, "gamma": 0.01}

    for rnd in range(rounds):
        print(f"\n--- Federated Round {rnd + 1} ---")
        queue = Queue()
        processes = []

        for i in range(num_clients):
            p = Process(target=client_worker,
                        args=(global_state, client_datasets[i], device_list[i],
                              dp_type, dp_params, accountant, epochs, queue, i))
            p.start()
            processes.append(p)

        local_states = [None] * num_clients
        for _ in range(num_clients):
            client_id, state = queue.get()
            local_states[client_id] = state

        for p in processes:
            p.join()

        global_state = federated_average(local_states)
        global_model.load_state_dict(global_state)
        acc = evaluate(global_model, testloader, device="cuda:0")
        eps_rem, delta_rem = accountant.remaining()
        print(f" Round {rnd + 1} Accuracy: {acc:.4f}, ε remaining: {eps_rem:.4f}, δ remaining: {delta_rem:.1e}")

        if eps_rem <= 0 or delta_rem <= 0:
            print("Privacy budget exhausted. Stopping training.")
            break

    print("Training complete.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    federated_training(rounds=3, epochs=1)