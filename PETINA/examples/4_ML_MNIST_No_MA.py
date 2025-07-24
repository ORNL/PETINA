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
    epochs      = 20
    sketch_rows = 5      
    sketch_cols = 520 #x10   
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
# Using Count Sketch rows=5, cols=520, blocks=1
# ===========================================

# === No DP Noise ===
#  Epoch 1 Accuracy: 0.9709                                                                                                                           
#  Epoch 2 Accuracy: 0.9855                                                                                                                           
#  Epoch 3 Accuracy: 0.9862                                                                                                                           
#  Epoch 4 Accuracy: 0.9881                                                                                                                           
#  Epoch 5 Accuracy: 0.9868                                                                                                                           
#  Epoch 6 Accuracy: 0.9876                                                                                                                           
#  Epoch 7 Accuracy: 0.9894                                                                                                                           
#  Epoch 8 Accuracy: 0.9880                                                                                                                           
#  Epoch 9 Accuracy: 0.9898                                                                                                                           
#  Epoch 10 Accuracy: 0.9895                                                                                                                          
#  Epoch 11 Accuracy: 0.9889                                                                                                                          
#  Epoch 12 Accuracy: 0.9895                                                                                                                          
#  Epoch 13 Accuracy: 0.9854                                                                                                                          
#  Epoch 14 Accuracy: 0.9906                                                                                                                          
#  Epoch 15 Accuracy: 0.9885                                                                                                                          
#  Epoch 16 Accuracy: 0.9906                                                                                                                          
#  Epoch 17 Accuracy: 0.9908                                                                                                                          
#  Epoch 18 Accuracy: 0.9893                                                                                                                          
#  Epoch 19 Accuracy: 0.9913                                                                                                                          
#  Epoch 20 Accuracy: 0.9902                                                                                                                          
# Training Done.
# Time run: 68.93 seconds

# === Gaussian DP Noise ===
#  Epoch 1 Accuracy: 0.9415                                                                                                                           
#  Epoch 2 Accuracy: 0.9625                                                                                                                           
#  Epoch 3 Accuracy: 0.9705                                                                                                                           
#  Epoch 4 Accuracy: 0.9711                                                                                                                           
#  Epoch 5 Accuracy: 0.9750                                                                                                                           
#  Epoch 6 Accuracy: 0.9781                                                                                                                           
#  Epoch 7 Accuracy: 0.9651                                                                                                                           
#  Epoch 8 Accuracy: 0.9803                                                                                                                           
#  Epoch 9 Accuracy: 0.9803                                                                                                                           
#  Epoch 10 Accuracy: 0.9771                                                                                                                          
#  Epoch 11 Accuracy: 0.9767                                                                                                                          
#  Epoch 12 Accuracy: 0.9774                                                                                                                          
#  Epoch 13 Accuracy: 0.9786                                                                                                                          
#  Epoch 14 Accuracy: 0.9752                                                                                                                          
#  Epoch 15 Accuracy: 0.9832                                                                                                                          
#  Epoch 16 Accuracy: 0.9800                                                                                                                          
#  Epoch 17 Accuracy: 0.9836                                                                                                                          
#  Epoch 18 Accuracy: 0.9829                                                                                                                          
#  Epoch 19 Accuracy: 0.9821                                                                                                                          
#  Epoch 20 Accuracy: 0.9819                                                                                                                          
# Training Done.
# Time run: 68.93 seconds

# === Laplace DP Noise ===
#  Epoch 1 Accuracy: 0.9636                                                                                                                           
#  Epoch 2 Accuracy: 0.9719                                                                                                                           
#  Epoch 3 Accuracy: 0.9812                                                                                                                           
#  Epoch 4 Accuracy: 0.9837                                                                                                                           
#  Epoch 5 Accuracy: 0.9869                                                                                                                           
#  Epoch 6 Accuracy: 0.9836                                                                                                                           
#  Epoch 7 Accuracy: 0.9851                                                                                                                           
#  Epoch 8 Accuracy: 0.9869                                                                                                                           
#  Epoch 9 Accuracy: 0.9849                                                                                                                           
#  Epoch 10 Accuracy: 0.9830                                                                                                                          
#  Epoch 11 Accuracy: 0.9881                                                                                                                          
#  Epoch 12 Accuracy: 0.9866                                                                                                                          
#  Epoch 13 Accuracy: 0.9869                                                                                                                          
#  Epoch 14 Accuracy: 0.9856                                                                                                                          
#  Epoch 15 Accuracy: 0.9877                                                                                                                          
#  Epoch 16 Accuracy: 0.9822                                                                                                                          
#  Epoch 17 Accuracy: 0.9866                                                                                                                          
#  Epoch 18 Accuracy: 0.9864                                                                                                                          
#  Epoch 19 Accuracy: 0.9872                                                                                                                          
#  Epoch 20 Accuracy: 0.9852                                                                                                                          
# Training Done.
# Time run: 68.98 seconds

# === CSVec + Gaussian DP ===
#  Epoch 1 Accuracy: 0.8857                                                                                                                           
#  Epoch 2 Accuracy: 0.9328                                                                                                                           
#  Epoch 3 Accuracy: 0.9556                                                                                                                           
#  Epoch 4 Accuracy: 0.9596                                                                                                                           
#  Epoch 5 Accuracy: 0.9684                                                                                                                           
#  Epoch 6 Accuracy: 0.9716                                                                                                                           
#  Epoch 7 Accuracy: 0.9686                                                                                                                           
#  Epoch 8 Accuracy: 0.9709                                                                                                                           
#  Epoch 9 Accuracy: 0.9741                                                                                                                           
#  Epoch 10 Accuracy: 0.9749                                                                                                                          
#  Epoch 11 Accuracy: 0.9754                                                                                                                          
#  Epoch 12 Accuracy: 0.9776                                                                                                                          
#  Epoch 13 Accuracy: 0.9756                                                                                                                          
#  Epoch 14 Accuracy: 0.9743                                                                                                                          
#  Epoch 15 Accuracy: 0.9754                                                                                                                          
#  Epoch 16 Accuracy: 0.9763                                                                                                                          
#  Epoch 17 Accuracy: 0.9776                                                                                                                          
#  Epoch 18 Accuracy: 0.9748                                                                                                                          
#  Epoch 19 Accuracy: 0.9757                                                                                                                          
#  Epoch 20 Accuracy: 0.9748                                                                                                                          
# Training Done.
# Time run: 69.19 seconds

# === CSVec + Laplace DP ===
#  Epoch 1 Accuracy: 0.9139                                                                                                                           
#  Epoch 2 Accuracy: 0.9504                                                                                                                           
#  Epoch 3 Accuracy: 0.9634                                                                                                                           
#  Epoch 4 Accuracy: 0.9690                                                                                                                           
#  Epoch 5 Accuracy: 0.9685                                                                                                                           
#  Epoch 6 Accuracy: 0.9711                                                                                                                           
#  Epoch 7 Accuracy: 0.9745                                                                                                                           
#  Epoch 8 Accuracy: 0.9731                                                                                                                           
#  Epoch 9 Accuracy: 0.9736                                                                                                                           
#  Epoch 10 Accuracy: 0.9759                                                                                                                          
#  Epoch 11 Accuracy: 0.9750                                                                                                                          
#  Epoch 12 Accuracy: 0.9753                                                                                                                          
#  Epoch 13 Accuracy: 0.9709                                                                                                                          
#  Epoch 14 Accuracy: 0.9784                                                                                                                          
#  Epoch 15 Accuracy: 0.9699                                                                                                                          
#  Epoch 16 Accuracy: 0.9767                                                                                                                          
#  Epoch 17 Accuracy: 0.9799                                                                                                                          
#  Epoch 18 Accuracy: 0.9785                                                                                                                          
#  Epoch 19 Accuracy: 0.9787                                                                                                                          
#  Epoch 20 Accuracy: 0.9772                                                                                                                          
# Training Done.
# Time run: 69.58 seconds