# ======================================================
#   MNIST Training with DP + Count Sketch + Bayesian Opt
# ======================================================
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# --- PETINA / Opacus Imports ---
from PETINA.package.Opacus_budget_accountant.accountants.gdp import GaussianAccountant
from PETINA.package.Opacus_budget_accountant.accountants.utils import get_noise_multiplier
from PETINA import DP_Mechanisms

# --- Bayesian Optimization ---
from PETINA.package.BayesianOptimization_local import BayesianOptimization

# =========================
# 1. Setup
# =========================
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
batch_size = 240
testbatchsize = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
testset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
dataset_size = len(trainset)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
testloader  = torch.utils.data.DataLoader(testset, batch_size=testbatchsize, shuffle=False, num_workers=2, pin_memory=True)

# =========================
# 2. Model
# =========================
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

# =========================
# 3. Evaluation
# =========================
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

# =========================
# 4. DP Wrappers
# =========================
def apply_gaussian_with_budget(grad: torch.Tensor, delta: float, epsilon: float, gamma: float) -> torch.Tensor:
    grad_np = grad.cpu().numpy()
    noisy_np = DP_Mechanisms.applyDPGaussian(grad_np, delta=delta, epsilon=epsilon, gamma=gamma)
    return torch.tensor(noisy_np, dtype=torch.float32).to(device)

# =========================
# 5. Training with DP + CSVec
# =========================
def train_model_with_budget(dp_type, dp_params, total_epoch=3, use_count_sketch=False, sketch_params=None):
    model = SampleConvNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.25, momentum=0)
    accountantOPC = GaussianAccountant()
    mechanism_map = {'gaussian': "gaussian"}

    for e in range(total_epoch):
        model.train()
        criterion = nn.CrossEntropyLoss()
        losses = []

        for data, target in tqdm(trainloader, desc=f"Epoch {e+1}", leave=False):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()

            if dp_type is not None:
                if use_count_sketch:
                    grad_list = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
                    if grad_list:
                        flat_grad = torch.cat(grad_list)
                        privatized_grad_tensor = DP_Mechanisms.applyCountSketch(
                            domain=flat_grad,
                            num_rows=sketch_params['d'],
                            num_cols=sketch_params['w'],
                            epsilon=dp_params['epsilon'],
                            delta=dp_params['delta'],
                            mechanism=mechanism_map[dp_type],
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
                else:
                    for p in model.parameters():
                        if p.grad is not None:
                            if dp_type == 'gaussian':
                                p.grad = apply_gaussian_with_budget(
                                    p.grad,
                                    delta=dp_params.get('delta', 1e-5),
                                    epsilon=dp_params.get('epsilon', 1.0),
                                    gamma=dp_params.get('gamma', 1.0)
                                )

                # Update accountant
                sample_rate = trainloader.batch_size / dataset_size
                sigma = get_noise_multiplier(
                    target_epsilon=dp_params['epsilon'],
                    target_delta=dp_params['delta'],
                    sample_rate=sample_rate,
                    epochs=total_epoch,
                    accountant="gdp",
                )
                accountantOPC.step(noise_multiplier=sigma, sample_rate=sample_rate)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        acc = evaluate(model, testloader)
        if dp_type is not None:
            epsilon_acc = accountantOPC.get_epsilon(delta=dp_params.get('delta', 1e-5))
            print(f"Epoch {e+1} Loss: {np.mean(losses):.6f} ε_accountant={epsilon_acc:.2f}, δ={dp_params.get('delta', 1e-5)} Test Acc={acc*100:.2f}%")
        else:
            print(f"Epoch {e+1} Loss: {np.mean(losses):.6f} Test Acc={acc*100:.2f}%")
    return acc

# =========================
# 6. Bayesian Optimization for Count Sketch
# =========================


def bayesian_opt_csvec(dp_params, total_epoch=3, n_calls=8):
    """
    Run Bayesian Optimization to find best CSVec sketch rows and cols
    using PETINA.package.BayesianOptimization_local
    """
    def objective(rows, cols):
        acc = train_model_with_budget(
            dp_type='gaussian',
            dp_params=dp_params,
            total_epoch=total_epoch,
            use_count_sketch=True,
            sketch_params={'d': int(rows), 'w': int(cols), 'numBlocks': 1}
        )
        # PETINA BO maximizes, so return accuracy directly
        return acc

    # Define search space
    search_space = {
        'rows': (3, 20),
        'cols': (100, 1000)
    }

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=search_space,
        random_state=42
    )
    
    optimizer.maximize(init_points=2, n_iter=n_calls-2)

    print("\n=== Bayesian Optimization Results ===")
    for i, res in enumerate(optimizer.res):
        print(f"| Iter {i+1} | Accuracy = {res['target']:.4f} | Rows = {res['params']['rows']:.1f} | Cols = {res['params']['cols']:.1f} |")
    
    best = optimizer.max
    print(f"\nBest config: Rows={best['params']['rows']:.1f}, Cols={best['params']['cols']:.1f}, Accuracy={best['target']:.4f}")
    return best

# =========================
# 7. Main
# =========================
def main():
    total_epoch = 5
    delta = 1e-5
    epsilon = 1
    gamma = 0.01

    # print("=========== DP Training Parameters ===========")
    # print(f"ε={epsilon}, δ={delta}, γ={gamma}, total epochs={total_epoch}")
    # print(f"Batch size: {batch_size}\n")

    # # === No DP Noise
    # print("=== Experiment 1: No DP ===")
    # train_model_with_budget(dp_type=None, dp_params={}, total_epoch=total_epoch)

    # # === Gaussian DP
    # print("\n=== Experiment 2: Gaussian DP ===")
    # train_model_with_budget(dp_type='gaussian', dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma}, total_epoch=total_epoch)

    # === CSVec + Gaussian DP with Bayesian Optimization
    print("\n=== Experiment 3: CSVec + Gaussian DP + Bayesian Optimization ===")
    bayesian_opt_csvec(dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma}, total_epoch=total_epoch, n_calls=8)

if __name__ == "__main__":
    main()
