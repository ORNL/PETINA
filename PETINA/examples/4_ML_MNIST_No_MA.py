# ======================================================
#        MNIST Training with Differential Privacy
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
testbatchsize=1024
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
        losses = []
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
# 6. Experiment Settings
# =========================
def main():
    delta       = 1e-5
    epsilon     = 1.0
    gamma       = 0.01
    sensitivity = 1.0
    epochs      = 15
    sketch_rows = 5       
    sketch_cols = 260   
    csvec_blocks = 1

    print("===== Differential Privacy Parameters =====")
    print(f"ε={epsilon}, δ={delta}, γ={gamma}, sensitivity={sensitivity}")
    print(f"Using Count Sketch rows={sketch_rows}, cols={sketch_cols}, blocks={csvec_blocks}")
    print("===========================================\n")

# =========================
# 7. Run Experiments with Timing
# =========================

    # === No DP ===
    print("=== No DP Noise ===")
    start = time.time()
    train_model(dp_type=None, dp_params={}, total_epochs=epochs)
    print(f"Time run: {time.time() - start:.2f} seconds\n")

    # # === Gaussian DP ===
    print("=== Gaussian DP Noise ===")
    start = time.time()
    train_model(dp_type='gaussian',
                dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma},
                total_epochs=epochs)
    print(f"Time run: {time.time() - start:.2f} seconds\n")

    # # === Laplace DP ===
    print("=== Laplace DP Noise ===")
    start = time.time()
    train_model(dp_type='laplace',
                dp_params={'sensitivity': sensitivity, 'epsilon': epsilon, 'gamma': gamma},
                total_epochs=epochs)
    print(f"Time run: {time.time() - start:.2f} seconds\n")

    # === CSVec + Gaussian DP ===
    print("=== CSVec + Gaussian DP ===")
    start = time.time()
    train_model(dp_type='gaussian',
                dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma},
                total_epochs=epochs,
                use_count_sketch=True,
                sketch_params={'d': sketch_rows, 'w': sketch_cols, 'numBlocks': csvec_blocks})
    print(f"Time run: {time.time() - start:.2f} seconds\n")

    # === CSVec + Laplace DP ===
    print("=== CSVec + Laplace DP ===")
    start = time.time()
    train_model(dp_type='laplace',
                dp_params={'sensitivity': sensitivity, 'epsilon': epsilon, 'gamma': gamma,'delta': delta},
                total_epochs=epochs,
                use_count_sketch=True,
                sketch_params={'d': sketch_rows, 'w': sketch_cols, 'numBlocks': csvec_blocks})
    print(f"Time run: {time.time() - start:.2f} seconds\n")

if __name__ == "__main__":
    main()
    
# --------------OUTPUT--------------
# ===== Differential Privacy Parameters =====
# ε=1.0, δ=1e-05, γ=0.01, sensitivity=1.0
# Using Count Sketch rows=5, cols=260, blocks=1
# ===========================================

# === No DP Noise ===
#  Epoch 1 Accuracy: 0.9663                                                                                          
#  Epoch 2 Accuracy: 0.9853                                                                                          
#  Epoch 3 Accuracy: 0.9874                                                                                          
#  Epoch 4 Accuracy: 0.9890                                                                                          
#  Epoch 5 Accuracy: 0.9861                                                                                          
#  Epoch 6 Accuracy: 0.9871                                                                                          
#  Epoch 7 Accuracy: 0.9898                                                                                          
#  Epoch 8 Accuracy: 0.9865                                                                                          
#  Epoch 9 Accuracy: 0.9896                                                                                          
#  Epoch 10 Accuracy: 0.9892                                                                                         
#  Epoch 11 Accuracy: 0.9885                                                                                         
#  Epoch 12 Accuracy: 0.9874                                                                                         
#  Epoch 13 Accuracy: 0.9895                                                                                         
#  Epoch 14 Accuracy: 0.9899                                                                                         
#  Epoch 15 Accuracy: 0.9875                                                                                         
# Training Done.
# Time run: 82.34 seconds

# === Gaussian DP Noise ===
#  Epoch 1 Accuracy: 0.9337                                                                                          
#  Epoch 2 Accuracy: 0.9582                                                                                          
#  Epoch 3 Accuracy: 0.9675                                                                                          
#  Epoch 4 Accuracy: 0.9737                                                                                          
#  Epoch 5 Accuracy: 0.9766                                                                                          
#  Epoch 6 Accuracy: 0.9769                                                                                          
#  Epoch 7 Accuracy: 0.9792                                                                                          
#  Epoch 8 Accuracy: 0.9777                                                                                          
#  Epoch 9 Accuracy: 0.9798                                                                                          
#  Epoch 10 Accuracy: 0.9800                                                                                         
#  Epoch 11 Accuracy: 0.9815                                                                                         
#  Epoch 12 Accuracy: 0.9747                                                                                         
#  Epoch 13 Accuracy: 0.9824                                                                                         
#  Epoch 14 Accuracy: 0.9810                                                                                         
#  Epoch 15 Accuracy: 0.9827                                                                                         
# Training Done.
# Time run: 87.34 seconds

# === Laplace DP Noise ===
#  Epoch 1 Accuracy: 0.9586                                                                                          
#  Epoch 2 Accuracy: 0.9766                                                                                          
#  Epoch 3 Accuracy: 0.9778                                                                                          
#  Epoch 4 Accuracy: 0.9842                                                                                          
#  Epoch 5 Accuracy: 0.9822                                                                                          
#  Epoch 6 Accuracy: 0.9853                                                                                          
#  Epoch 7 Accuracy: 0.9854                                                                                          
#  Epoch 8 Accuracy: 0.9857                                                                                          
#  Epoch 9 Accuracy: 0.9866                                                                                          
#  Epoch 10 Accuracy: 0.9857                                                                                         
#  Epoch 11 Accuracy: 0.9868                                                                                         
#  Epoch 12 Accuracy: 0.9848                                                                                         
#  Epoch 13 Accuracy: 0.9865                                                                                         
#  Epoch 14 Accuracy: 0.9838                                                                                         
#  Epoch 15 Accuracy: 0.9869                                                                                         
# Training Done.
# Time run: 84.64 seconds

# === CSVec + Gaussian DP ===
#  Epoch 1 Accuracy: 0.8686                                                                                          
#  Epoch 2 Accuracy: 0.9300                                                                                          
#  Epoch 3 Accuracy: 0.9389                                                                                          
#  Epoch 4 Accuracy: 0.9536                                                                                          
#  Epoch 5 Accuracy: 0.9538                                                                                          
#  Epoch 6 Accuracy: 0.9604                                                                                          
#  Epoch 7 Accuracy: 0.9619                                                                                          
#  Epoch 8 Accuracy: 0.9679                                                                                          
#  Epoch 9 Accuracy: 0.9620                                                                                          
#  Epoch 10 Accuracy: 0.9642                                                                                         
#  Epoch 11 Accuracy: 0.9607                                                                                         
#  Epoch 12 Accuracy: 0.9707                                                                                         
#  Epoch 13 Accuracy: 0.9665                                                                                         
#  Epoch 14 Accuracy: 0.9693                                                                                         
#  Epoch 15 Accuracy: 0.9665                                                                                         
# Training Done.
# Time run: 86.91 seconds

# === CSVec + Laplace DP ===
#  Epoch 1 Accuracy: 0.8829                                                                                          
#  Epoch 2 Accuracy: 0.9126                                                                                          
#  Epoch 3 Accuracy: 0.9330                                                                                          
#  Epoch 4 Accuracy: 0.9508                                                                                          
#  Epoch 5 Accuracy: 0.9509                                                                                          
#  Epoch 6 Accuracy: 0.9546                                                                                          
#  Epoch 7 Accuracy: 0.9538                                                                                          
#  Epoch 8 Accuracy: 0.9598                                                                                          
#  Epoch 9 Accuracy: 0.9638                                                                                          
#  Epoch 10 Accuracy: 0.9585                                                                                         
#  Epoch 11 Accuracy: 0.9584                                                                                         
#  Epoch 12 Accuracy: 0.9596                                                                                         
#  Epoch 13 Accuracy: 0.9591                                                                                         
#  Epoch 14 Accuracy: 0.9624                                                                                         
#  Epoch 15 Accuracy: 0.9621                                                                                         
# Training Done.
# Time run: 79.04 seconds