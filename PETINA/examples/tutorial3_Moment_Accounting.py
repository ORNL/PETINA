
import torch
import torch.nn.functional as F
import numpy as np
import warnings
from PETINA import CSVec

# --- New Custom Exception ---
class BudgetExceededError(Exception):
    """Custom exception raised when the privacy budget is exceeded."""
    pass

# --- BudgetAccountant Class with new methods ---
class BudgetAccountant:
    """
    Tracks and manages the privacy budget (epsilon and delta) spent.
    """
    def __init__(self, total_epsilon, total_delta):
        """
        Initializes the budget accountant with a total privacy budget.

        Args:
            total_epsilon (float): The total epsilon budget available.
            total_delta (float): The total delta budget available.
        """
        if total_epsilon <= 0:
            raise ValueError("Total epsilon budget must be a positive value.")
        if total_delta < 0 or total_delta >= 1:
            raise ValueError("Total delta budget must be between 0 and 1.")

        self._total_epsilon = total_epsilon
        self._total_delta = total_delta
        self._spent_epsilon = 0.0
        self._spent_delta = 0.0
        self._spent_history = []

    def get_remaining_budget(self):
        """
        Returns the remaining privacy budget.

        Returns:
            tuple: A tuple containing the remaining (epsilon, delta).
        """
        return (self._total_epsilon - self._spent_epsilon, self._total_delta - self._spent_delta)

    def get_spent_budget(self):
        """
        Returns the total privacy budget spent so far.

        Returns:
            tuple: A tuple containing the spent (epsilon, delta).
        """
        return (self._spent_epsilon, self._spent_delta)
    
    def get_spent_history(self):
        """
        Returns a list of all individual privacy spends.

        Returns:
            list: A list of tuples (epsilon_spent, delta_spent, mechanism_name).
        """
        return self._spent_history

    def can_spend(self, epsilon_cost, delta_cost):
        """
        Checks if a given spend is possible without exceeding the budget.
        Does NOT raise an exception.

        Args:
            epsilon_cost (float): Epsilon cost to check.
            delta_cost (float): Delta cost to check.

        Returns:
            bool: True if the budget can be spent, False otherwise.
        """
        # Add a small tolerance for floating-point errors, especially for delta.
        return (self._spent_epsilon + epsilon_cost <= self._total_epsilon) and \
               (self._spent_delta + delta_cost <= self._total_delta + 1e-9)

    def _check_budget_expenditure(self, epsilon_cost, delta_cost):
        """
        Checks if spending the given budget will exceed the total budget.
        Raises BudgetExceededError if the budget is exceeded.
        """
        if not self.can_spend(epsilon_cost, delta_cost):
            # Raise the custom exception with a detailed message
            if self._spent_epsilon + epsilon_cost > self._total_epsilon:
                raise BudgetExceededError(
                    f"Epsilon budget exceeded! "
                    f"Remaining: {self.get_remaining_budget()[0]:.4f}, "
                    f"Attempted spend: {epsilon_cost:.4f}, "
                    f"Total spent: {self._spent_epsilon:.4f}, "
                    f"Total budget: {self._total_epsilon:.4f}"
                )
            if self._spent_delta + delta_cost > self._total_delta + 1e-9:
                raise BudgetExceededError(
                    f"Delta budget exceeded! "
                    f"Remaining: {self.get_remaining_budget()[1]:.10f}, "
                    f"Attempted spend: {delta_cost:.10f}, "
                    f"Total spent: {self._spent_delta:.10f}, "
                    f"Total budget: {self._total_delta:.10f}"
                )

    def spend(self, epsilon_cost, delta_cost, mechanism_name=""):
        """
        Records a privacy spend and updates the spent budget.

        Args:
            epsilon_cost (float): Epsilon cost to spend.
            delta_cost (float): Delta cost to spend.
            mechanism_name (str): The name of the mechanism being used.
        """
        self._check_budget_expenditure(epsilon_cost, delta_cost)
        
        self._spent_epsilon += epsilon_cost
        self._spent_delta += delta_cost
        self._spent_history.append((epsilon_cost, delta_cost, mechanism_name))

        print(f"Budget spent: ({epsilon_cost:.4f}, {delta_cost:.10f}) for '{mechanism_name}'.")
        print(f"Remaining budget: ({self.get_remaining_budget()[0]:.4f}, {self.get_remaining_budget()[1]:.10f})\n")

# --- Modified DP Functions to use the BudgetAccountant ---
def get_l1_sensitivity(tensor):
    """Calculates the L1 sensitivity of a tensor."""
    return torch.norm(tensor, p=1).item()

def get_l2_sensitivity(tensor):
    """Calculates the L2 sensitivity of a tensor."""
    return torch.norm(tensor, p=2).item()

def applyDPLaplace(tensor, epsilon, accountant=None, sensitivity=1.0):
    """
    Adds Laplace noise and optionally spends budget from an accountant.
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=scale, size=tensor.shape)
    
    # Check and spend budget if an accountant is provided
    if accountant:
        # Laplace mechanism is (epsilon, 0)-DP
        cost_epsilon, cost_delta = epsilon, 0.0
        # The spend method will now raise the custom exception if it fails
        accountant.spend(cost_epsilon, cost_delta, mechanism_name="Laplace Noise")
    
    if isinstance(tensor, torch.Tensor):
        return tensor + torch.tensor(noise, dtype=tensor.dtype)
    return tensor + noise

def applyDPGaussian(tensor, epsilon, delta, accountant=None, sensitivity=1.0):
    """
    Adds Gaussian noise and optionally spends budget from an accountant.
    """
    sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
    noise = np.random.normal(loc=0, scale=sigma, size=tensor.shape)

    if accountant:
        cost_epsilon, cost_delta = epsilon, delta
        accountant.spend(cost_epsilon, cost_delta, mechanism_name="Gaussian Noise")
    
    if isinstance(tensor, torch.Tensor):
        return tensor + torch.tensor(noise, dtype=tensor.dtype)
    return tensor + noise

def applyClipping(tensor, max_norm):
    """Clips the L2 norm of the tensor to a maximum value."""
    norm = torch.norm(tensor, p=2)
    if norm > max_norm:
        tensor = tensor * (max_norm / norm)
    return tensor

def applyCountSketch(items, num_rows, num_cols):
    """Applies the Count Sketch algorithm."""
    if isinstance(items, list):
        items_tensor = torch.tensor(items, dtype=torch.float32)
    elif isinstance(items, np.ndarray):
        items_tensor = torch.from_numpy(items).float()
    elif isinstance(items, torch.Tensor):
        items_tensor = items.float()
    else:
        raise TypeError("Input items must be a list, numpy array, or torch tensor.")
    dimension = items_tensor.numel()
    cs_vec = CSVec(d=dimension, c=num_cols, r=num_rows)
    cs_vec.accumulateVec(items_tensor)
    return cs_vec

def torch_to_list(tensor):
    """Converts a torch tensor to a flattened Python list."""
    return tensor.flatten().tolist()

def list_to_numpy(data_list):
    """Converts a list to a numpy array."""
    return np.array(data_list)

def getModelDimension(model):
    """Calculates the total number of parameters in a model."""
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params

def add_noise_to_update(update_tensor, noise_multiplier):
    """Adds Gaussian noise to a tensor."""
    noise = torch.randn_like(update_tensor) * noise_multiplier
    return update_tensor + noise

# --- Modified client_update with graceful error handling ---
def client_update(client_model, optimizer, train_loader, epoch=5, use_privacy=False, privacy_method=None, clipping_norm=1.0, noise_multiplier=0.0, accountant=None):
    """
    Performs local training on a client's model with graceful budget tracking.
    """
    client_model.train()
    
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            # --- Privacy and Clipping ---
            if use_privacy:
                # 1. Apply clipping first to bound the sensitivity
                torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=clipping_norm)
                
                # 2. Apply noise and track budget with a try-except block
                try:
                    # We will calculate the cost and attempt to spend it for each parameter
                    for param in client_model.parameters():
                        if param.grad is not None:
                            if privacy_method == 'laplace':
                                # Note: `noise_multiplier` is used as `epsilon` here.
                                applyDPLaplace(param.grad, noise_multiplier, accountant=accountant, sensitivity=clipping_norm)
                                
                            elif privacy_method == 'gaussian':
                                # This calculation assumes a fixed delta and clips the gradients
                                delta_for_cost = accountant.get_remaining_budget()[1] / (epoch * len(train_loader) * getModelDimension(client_model))
                                if delta_for_cost <= 0:
                                    raise BudgetExceededError("Delta budget is zero or negative.")
                                
                                # This `applyDPGaussian` calculates the noise scale from epsilon/delta.
                                # Let's assume noise_multiplier is sigma and we calculate epsilon.
                                # This is a common pattern for tracking in DP-SGD.
                                sigma = noise_multiplier
                                epsilon_cost = (clipping_norm / sigma) * np.sqrt(2 * np.log(1.25 / delta_for_cost))
                                
                                # Use the original noise function for consistency with your code.
                                param.grad.copy_(add_noise_to_update(param.grad, noise_multiplier))
                                
                                # Now, account for the spend.
                                accountant.spend(epsilon_cost, delta_for_cost, mechanism_name="Gaussian Noise (per batch)")

                except BudgetExceededError as e:
                    # Gracefully catch the error and break the training loop
                    print(f"\n--- Budget Exhausted! Stopping local training gracefully. ---")
                    print(f"Reason: {e}")
                    return loss.item() # Exit the function early

            # This step only runs if the budget was not exhausted
            optimizer.step()

    return loss.item()

# --- Example Usage ---

# First, let's create a dummy model and data loader for the example.
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Dummy model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Dummy data
dummy_data = torch.randn(100, 10)
dummy_labels = torch.randint(0, 2, (100,))
dummy_dataset = TensorDataset(dummy_data, dummy_labels)
dummy_loader = DataLoader(dummy_dataset, batch_size=10)

print("--- Example Usage of Graceful Budget Catching ---")

# 1. Initialize the Budget Accountant with a tight budget
total_epsilon = 100
total_delta = 1e-5
accountant = BudgetAccountant(total_epsilon=total_epsilon, total_delta=total_delta)

print(f"Total budget initialized: Epsilon = {accountant._total_epsilon}, Delta = {accountant._total_delta}")
print(f"Initial remaining budget: {accountant.get_remaining_budget()}\n")

# 2. Run training with Laplace noise that will exceed the budget
print("Starting training with Laplace noise (expecting budget exhaustion)...")
client_model_laplace = SimpleModel()
optimizer_laplace = torch.optim.SGD(client_model_laplace.parameters(), lr=0.01)

# We have 10 batches in total. We will spend 0.1 epsilon per parameter per batch.
# Total parameters is 62. So total spend per batch is 6.2 epsilon.
# Our total budget is 1.0. The budget will be exhausted on the very first batch.
_ = client_update(
    client_model=client_model_laplace,
    optimizer=optimizer_laplace,
    train_loader=dummy_loader,
    epoch=1,
    use_privacy=True,
    privacy_method='laplace',
    clipping_norm=1.0, 
    noise_multiplier=0.1, # This is the epsilon for the Laplace mechanism per parameter
    accountant=accountant
)

print("\n--- Training has finished or exited gracefully. ---")
print(f"Final spent budget: ({accountant.get_spent_budget()[0]:.4f}, {accountant.get_spent_budget()[1]:.10f})")
print(f"Final remaining budget: ({accountant.get_remaining_budget()[0]:.4f}, {accountant.get_remaining_budget()[1]:.10f})")
print(f"\nSpending History:")
for spend in accountant.get_spent_history():
    print(f"  - Mechanism: {spend[2]}, Epsilon Cost: {spend[0]:.4f}, Delta Cost: {spend[1]:.10f}")


# 3. Proactive Budget Check Example
print("\n--- Example of Proactive Budget Checking with `can_spend` ---")
new_accountant = BudgetAccountant(total_epsilon=0.5, total_delta=1e-5)
print(f"New budget: {new_accountant.get_remaining_budget()}")
epsilon_cost_per_batch = 0.05
delta_cost_per_batch = 0

for batch_idx, (data, target) in enumerate(dummy_loader):
    # Check if we can spend the budget for this batch before doing anything
    # This is a good place to do a check for the entire batch
    # total_cost_for_batch = epsilon_cost_per_batch * num_parameters
    # We will just check for one parameter for simplicity
    
    if not new_accountant.can_spend(epsilon_cost_per_batch, delta_cost_per_batch):
        print(f"\nBudget exhausted before batch {batch_idx+1}. Stopping training.")
        break # Exit the batch loop gracefully
    
    # If we can spend, proceed with the operation
    print(f"Processing batch {batch_idx+1}...")
    
    # Simulate spending the budget for this batch
    new_accountant.spend(epsilon_cost_per_batch, delta_cost_per_batch, mechanism_name="Laplace Noise (Proactive)")

print(f"\nFinal remaining budget from proactive check: {new_accountant.get_remaining_budget()}")