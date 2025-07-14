import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PETINA import DP_Mechanisms  # Assuming applyDPLaplace is here

# --- Setup device ---
# Check if ROCm is available (PyTorch with ROCm)
print(torch.cuda.get_device_name(0))   # Should print something like "NVIDIA A100"
print(torch.cuda.current_device())     # GPU index
print(torch.cuda.device_count())  
if torch.cuda.is_available():
    device = torch.device("cuda")  # ROCm PyTorch uses "cuda" device string for AMD GPU
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Rest of your code (model, dataloaders, training) unchanged


# --- Load CIFAR-10 dataset ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 32x16x16
        x = self.pool(self.relu(self.conv2(x)))  # 64x8x8
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# --- DP Parameters ---
sensitivity = 1.0
epsilon = 0.3  # Privacy budget per batch
rounds = 2
epochs_per_round = 3

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

print("Training CIFAR-10 with PETINA DP noise, rounds and epochs...")

for r in range(1, rounds + 1):
    print(f"\n--- Round {r} ---")
    for epoch in range(1, epochs_per_round + 1):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Add Laplace noise to gradients using PETINA
            for param in model.parameters():
                if param.grad is not None:
                    grad_np = param.grad.detach().cpu().numpy()
                    noisy_grad = DP_Mechanisms.applyDPLaplace(grad_np, sensitivity, epsilon)
                    param.grad.data = torch.tensor(noisy_grad, dtype=param.grad.dtype).to(device)

            optimizer.step()
            print(f"Round {r} Epoch {epoch} Batch {batch_idx} done")

        acc = evaluate(model, testloader)
        print(f"Round {r} Epoch {epoch} test accuracy: {acc:.4f}")

print("Training with DP completed.")
