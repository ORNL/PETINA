# ======================================================
#        MNIST Training with Differential Privacy + Bayesian Opt
# ======================================================
import random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# --- PETINA Imports ---
from PETINA import DP_Mechanisms

# --- Bayesian Optimization Imports ---
from PETINA.package.BayesianOptimization_local import BayesianOptimization
from PETINA.package.BayesianOptimization_local.util import UtilityFunction

import pandas as pd

# =========================
# 1. Setup and Utilities
# =========================
def set_seed(seed=42):
    """Ensure reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. Load MNIST Dataset
# =========================
batch_size = 240 
testbatchsize = 1024
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)) 
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
testset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
testloader  = torch.utils.data.DataLoader(testset, batch_size=testbatchsize, shuffle=False, num_workers=2, pin_memory=True)

# =========================
# 3. Define Simple CNN
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
# 4. Evaluation Function
# =========================
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    return correct / total

# =========================
# 5. Training Function
# =========================
def train_model(dp_type=None, dp_params=None, total_epochs=3,
                use_count_sketch=False, sketch_params=None):
    mechanism_map = {
        'gaussian': "gaussian",
        'laplace': "laplace"
    }
    model = SampleConvNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.25, momentum=0)
    for e in range(total_epochs):
        model.train()
        criterion = nn.CrossEntropyLoss()
        progress = tqdm(trainloader, desc=f"Epoch {e + 1}", leave=False)
        for data, target in progress:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            if dp_type is not None:
                if use_count_sketch:
                    grad_list = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
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
                else:
                    for p in model.parameters():
                        if dp_type == 'gaussian':
                            p.grad = DP_Mechanisms.applyDPGaussian(
                                p.grad,
                                delta=dp_params.get('delta', 1e-5),
                                epsilon=dp_params.get('epsilon', 1.0),
                                gamma=dp_params.get('gamma', 1.0)
                            ).to(p.grad.device)
                        elif dp_type == 'laplace':
                            p.grad = DP_Mechanisms.applyDPLaplace(
                                p.grad, 
                                sensitivity=dp_params.get('sensitivity', 1.0), 
                                epsilon=dp_params.get('epsilon', 1.0), 
                                gamma=dp_params.get('gamma', 1.0)
                            ).to(p.grad.device)
                        else:
                            raise ValueError(f"Unknown dp_type: {dp_type}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            progress.set_postfix(loss=loss.item())
        acc = evaluate(model, testloader)
        print(f" Epoch {e + 1} Accuracy: {acc:.4f}")
    print("Training Done.")
    return model

# =========================
# 6. Black-box for Bayesian Optimization
# =========================
def black_box(sketch_rows, sketch_cols,
              dp_type='gaussian',
              dp_params=None,
              eval_epochs=3,
              cs_blocks=1):
    sketch_rows = int(max(1, round(sketch_rows)))
    sketch_cols = int(max(1, round(sketch_cols)))
    if dp_params is None:
        dp_params = {'delta': 1e-5, 'epsilon': 1.0, 'gamma': 0.01}
    model = train_model(dp_type=dp_type,
                        dp_params=dp_params,
                        total_epochs=eval_epochs,
                        use_count_sketch=True,
                        sketch_params={'d': sketch_rows, 'w': sketch_cols, 'numBlocks': cs_blocks})
    final_acc = evaluate(model, testloader)
    return float(final_acc)

def run_bayesian_optimization(dp_type='gaussian',
                              dp_params=None,
                              pbounds={"sketch_rows": (2, 16), "sketch_cols": (64, 1024)},
                              init_points=4,
                              n_iter=12,
                              eval_epochs=3,
                              cs_blocks=1,
                              save_csv="bayes_sketch_results.csv"):
    def _target(sketch_rows, sketch_cols):
        return black_box(sketch_rows=sketch_rows,
                         sketch_cols=sketch_cols,
                         dp_type=dp_type,
                         dp_params=dp_params,
                         eval_epochs=eval_epochs,
                         cs_blocks=cs_blocks)
    optimizer = BayesianOptimization(f=_target, pbounds=pbounds, verbose=2, random_state=42)
    utility = UtilityFunction(kind="ucb", kappa=1.96, xi=0.01)
    records = []
    optimizer.maximize(init_points=init_points, n_iter=0)
    for i in range(n_iter):
        next_point = optimizer.suggest(utility)
        next_point['sketch_rows'] = int(round(next_point['sketch_rows']))
        next_point['sketch_cols'] = int(round(next_point['sketch_cols']))
        print(f"\nTrial {i+1}/{n_iter}: rows={next_point['sketch_rows']} cols={next_point['sketch_cols']}")
        t0 = time.time()
        target = _target(next_point['sketch_rows'], next_point['sketch_cols'])
        t_elapsed = time.time() - t0
        print(f" -> accuracy={target:.4f} (time {t_elapsed:.1f}s)")
        try:
            optimizer.register(params=next_point, target=target)
        except Exception:
            pass
        records.append({
            'sketch_rows': next_point['sketch_rows'],
            'sketch_cols': next_point['sketch_cols'],
            'accuracy': target,
            'time_s': t_elapsed
        })
        pd.DataFrame(records).to_csv(save_csv, index=False)
    print("Best result:", optimizer.max)
    return optimizer, pd.DataFrame(records)

# =========================
# 7. Experiment Settings & Run
# =========================
def main():
    delta       = 1e-5
    epsilon     = 1.0
    gamma       = 0.01
    sensitivity = 1.0
    epochs      = 20
    sketch_rows = 5      
    sketch_cols = 520   
    csvec_blocks = 1

    print("===== Differential Privacy Parameters =====")
    print(f"ε={epsilon}, δ={delta}, γ={gamma}, sensitivity={sensitivity}")
    print(f"Using Count Sketch rows={sketch_rows}, cols={sketch_cols}, blocks={csvec_blocks}")
    print("===========================================\n")

    # # === No DP ===
    # print("=== No DP Noise ===")
    # start = time.time()
    # train_model(dp_type=None, dp_params={}, total_epochs=epochs)
    # print(f"Time run: {time.time() - start:.2f} seconds\n")

    # # === Gaussian DP ===
    # print("=== Gaussian DP Noise ===")
    # start = time.time()
    # train_model(dp_type='gaussian',
    #             dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma},
    #             total_epochs=epochs)
    # print(f"Time run: {time.time() - start:.2f} seconds\n")

    # # === Laplace DP ===
    # print("=== Laplace DP Noise ===")
    # start = time.time()
    # train_model(dp_type='laplace',
    #             dp_params={'sensitivity': sensitivity, 'epsilon': epsilon, 'gamma': gamma},
    #             total_epochs=epochs)
    # print(f"Time run: {time.time() - start:.2f} seconds\n")

    # # === CSVec + Gaussian DP ===
    # print("=== CSVec + Gaussian DP ===")
    # start = time.time()
    # train_model(dp_type='gaussian',
    #             dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma},
    #             total_epochs=epochs,
    #             use_count_sketch=True,
    #             sketch_params={'d': sketch_rows, 'w': sketch_cols, 'numBlocks': csvec_blocks})
    # print(f"Time run: {time.time() - start:.2f} seconds\n")

    # # === CSVec + Laplace DP ===
    # print("=== CSVec + Laplace DP ===")
    # start = time.time()
    # train_model(dp_type='laplace',
    #             dp_params={'sensitivity': sensitivity, 'epsilon': epsilon, 'gamma': gamma,'delta': delta},
    #             total_epochs=epochs,
    #             use_count_sketch=True,
    #             sketch_params={'d': sketch_rows, 'w': sketch_cols, 'numBlocks': csvec_blocks})
    # print(f"Time run: {time.time() - start:.2f} seconds\n")

    # === Optional: Bayesian Optimization ===
    print("=== Bayesian Optimization for Count Sketch ===")
    dp_params = {'delta': delta, 'epsilon': epsilon, 'gamma': gamma}
    pbounds = {"sketch_rows": (3, 13), "sketch_cols": (64, 1024)}
    optimizer, history_df = run_bayesian_optimization(dp_type='gaussian',
                                                      dp_params=dp_params,
                                                      pbounds=pbounds,
                                                      init_points=4,
                                                      n_iter=8,
                                                      eval_epochs=3,
                                                      cs_blocks=csvec_blocks,
                                                      save_csv="bayes_sketch_results.csv")
    print("Bayes Optimization History (head):")
    print(history_df.head())

if __name__ == "__main__":
    main()
