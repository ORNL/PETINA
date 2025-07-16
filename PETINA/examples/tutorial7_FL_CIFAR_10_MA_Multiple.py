import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import copy
from PETINA import DP_Mechanisms, BudgetAccountant, BudgetError
from concurrent.futures import ThreadPoolExecutor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

num_clients = 3
client_data_size = len(trainset) // num_clients
split_sizes = [client_data_size] * (num_clients - 1)
split_sizes.append(len(trainset) - sum(split_sizes))
client_datasets = random_split(trainset, split_sizes)

def get_loader(dataset):
    return DataLoader(dataset, batch_size=64, shuffle=True)

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

def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

def local_train(model, dataloader, dp_type, dp_params, accountant, epochs=1, client_id=None):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Client {client_id+1} Epoch {epoch+1}", leave=False)
        for inputs, targets in pbar:
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
                        epsilon=dp_params['epsilon'],
                        gamma=dp_params['gamma'],
                        accountant=accountant
                    )
                elif dp_type == 'gaussian':
                    noisy_grad = DP_Mechanisms.applyDPGaussian(
                        grad, delta=dp_params['delta'],
                        epsilon=dp_params['epsilon'],
                        gamma=dp_params['gamma'],
                        accountant=accountant
                    )
                else:
                    noisy_grad = grad
                p.grad = torch.tensor(noisy_grad, dtype=p.grad.dtype).to(device)
            optimizer.step()
    return model

def federated_average(models):
    global_model = copy.deepcopy(models[0])
    for key in global_model.state_dict().keys():
        avg = torch.stack([m.state_dict()[key].float() for m in models], 0).mean(0)
        global_model.state_dict()[key].copy_(avg)
    return global_model

def federated_training(rounds=3, epochs_per_client=1):
    global_model = SimpleCNN().to(device)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)

    accountant = BudgetAccountant(epsilon=5.0, delta=1)
    dp_type = "gaussian"
    dp_params = {"delta": 1e-6, "epsilon": 0.0001, "gamma": 0.01}

    for rnd in range(rounds):
        print(f"\n--- Federated Round {rnd + 1} ---")
        local_models = []

        try:
            with ThreadPoolExecutor(max_workers=num_clients) as executor:
                futures = []
                for client_id in range(num_clients):
                    client_model = copy.deepcopy(global_model).to(device)
                    loader = get_loader(client_datasets[client_id])
                    futures.append(executor.submit(local_train, client_model, loader, dp_type, dp_params, accountant, epochs_per_client, client_id))
                for f in futures:
                    local_models.append(f.result())

            global_model = federated_average(local_models)
            acc = evaluate(global_model, testloader)
            eps_rem, delta_rem = accountant.remaining()
            print(f" Round {rnd + 1} Accuracy: {acc:.4f}, ε remaining: {eps_rem:.4f}, δ remaining: {delta_rem:.1e}")

            if eps_rem <= 0 or delta_rem <= 0:
                print("Privacy budget exhausted. Stopping training.")
                break

        except BudgetError as be:
            print(f"BudgetError: {be}")
            break

    print("Training complete.")
    return global_model

if __name__ == "__main__":
    federated_training()