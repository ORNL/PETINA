import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from PETINA import BudgetAccountant, BudgetError, DP_Mechanisms
import time

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

batch_size = 128
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

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
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    return correct / total

# --- DP noise wrappers with budget accounting ---
def apply_laplace_with_budget(grad, sensitivity, epsilon, gamma, accountant):
    """Apply Laplace noise to grad and spend budget."""
    grad_np = grad.cpu().numpy()
    noisy = DP_Mechanisms.applyDPLaplace(grad_np, sensitivity=sensitivity, epsilon=epsilon, gamma=gamma, accountant=accountant)
    return torch.tensor(noisy, dtype=torch.float32).to(device)

def apply_gaussian_with_budget(grad, delta, epsilon, gamma, accountant):
    """Apply Gaussian noise to grad and spend budget."""
    grad_np = grad.cpu().numpy()
    noisy = DP_Mechanisms.applyDPGaussian(grad_np, delta=delta, epsilon=epsilon, gamma=gamma, accountant=accountant)
    return torch.tensor(noisy, dtype=torch.float32).to(device)

# --- Training with DP and budget accounting ---
def train_model_with_budget(dp_type, dp_params, total_epsilon, total_delta, rounds=2, epochs_per_round=3):
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Initialize budget accountant
    accountant = BudgetAccountant(epsilon=total_epsilon, delta=total_delta)
    print(f"Initialized BudgetAccountant: ε={total_epsilon}, δ={total_delta}")

    try:
        for r in range(rounds):
            print(f"\nRound {r + 1}/{rounds}")
            for e in range(epochs_per_round):
                model.train()
                progress_bar = tqdm(trainloader, desc=f"Epoch {e + 1}", leave=False)
                for inputs, targets in progress_bar:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()

                    # Apply DP noise and spend budget
                    if dp_type is not None:
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

                    optimizer.step()
                    progress_bar.set_postfix(loss=loss.item(), eps_rem=accountant.remaining()[0])

                acc = evaluate(model, testloader)
                print(f" Epoch {e + 1} Test Accuracy: {acc:.4f}")
                eps_rem, delta_rem = accountant.remaining()
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
    total_epsilon = 10
    total_delta = 1

    rounds = 2
    epochs_per_round = 3

    print("\n=== Experiment 1: No DP Noise ===")
    train_model_with_budget(dp_type=None, dp_params={}, total_epsilon=total_epsilon, total_delta=total_delta,
                            rounds=rounds, epochs_per_round=epochs_per_round)

    print("\n=== Experiment 2: Gaussian DP Noise with Budget Accounting ===")
    train_model_with_budget(dp_type='gaussian',
                            dp_params={'delta': 1e-5, 'epsilon': 0.0005, 'gamma': 0.01},
                            total_epsilon=total_epsilon, total_delta=total_delta,
                            rounds=rounds, epochs_per_round=epochs_per_round)

    print("\n=== Experiment 3: Laplace DP Noise with Budget Accounting ===")
    train_model_with_budget(dp_type='laplace',
                            dp_params={'sensitivity': 1.0, 'epsilon': 0.0005, 'gamma': 0.01},
                            total_epsilon=total_epsilon, total_delta=0.0,
                            rounds=rounds, epochs_per_round=epochs_per_round)
