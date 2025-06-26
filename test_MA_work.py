# import torch
# import torch.nn.functional as F
# import numpy as np
# import warnings
# from csvec.csvec import CSVec

# # --- BudgetAccountant Class ---
# class BudgetAccountant:
#     """
#     Tracks and manages the privacy budget (epsilon and delta) spent.
#     """
#     def __init__(self, total_epsilon, total_delta):
#         """
#         Initializes the budget accountant with a total privacy budget.

#         Args:
#             total_epsilon (float): The total epsilon budget available.
#             total_delta (float): The total delta budget available.
#         """
#         if total_epsilon <= 0:
#             raise ValueError("Total epsilon budget must be a positive value.")
#         if total_delta < 0 or total_delta >= 1:
#             raise ValueError("Total delta budget must be between 0 and 1.")

#         self._total_epsilon = total_epsilon
#         self._total_delta = total_delta
#         self._spent_epsilon = 0.0
#         self._spent_delta = 0.0
#         self._spent_history = []

#     def get_remaining_budget(self):
#         """
#         Returns the remaining privacy budget.

#         Returns:
#             tuple: A tuple containing the remaining (epsilon, delta).
#         """
#         return (self._total_epsilon - self._spent_epsilon, self._total_delta - self._spent_delta)

#     def get_spent_budget(self):
#         """
#         Returns the total privacy budget spent so far.

#         Returns:
#             tuple: A tuple containing the spent (epsilon, delta).
#         """
#         return (self._spent_epsilon, self._spent_delta)
    
#     def get_spent_history(self):
#         """
#         Returns a list of all individual privacy spends.

#         Returns:
#             list: A list of tuples (epsilon_spent, delta_spent, mechanism_name).
#         """
#         return self._spent_history

#     def _check_budget_expenditure(self, epsilon_cost, delta_cost):
#         """
#         Checks if spending the given budget will exceed the total budget.

#         Args:
#             epsilon_cost (float): Epsilon cost to check.
#             delta_cost (float): Delta cost to check.

#         Raises:
#             RuntimeError: If the budget would be exceeded.
#         """
#         if self._spent_epsilon + epsilon_cost > self._total_epsilon:
#             raise RuntimeError(
#                 f"Epsilon budget exceeded! "
#                 f"Remaining: {self.get_remaining_budget()[0]:.4f}, "
#                 f"Attempted spend: {epsilon_cost:.4f}, "
#                 f"Total spent: {self._spent_epsilon:.4f}, "
#                 f"Total budget: {self._total_epsilon:.4f}"
#             )
#         if self._spent_delta + delta_cost > self._total_delta:
#             # Delta is usually tiny, so we check for strict inequality.
#             # It's more about preventing the sum from exceeding the max delta.
#             if self._spent_delta + delta_cost > self._total_delta + 1e-9: # Add a small tolerance for floating-point errors
#                 raise RuntimeError(
#                     f"Delta budget exceeded! "
#                     f"Remaining: {self.get_remaining_budget()[1]:.10f}, "
#                     f"Attempted spend: {delta_cost:.10f}, "
#                     f"Total spent: {self._spent_delta:.10f}, "
#                     f"Total budget: {self._total_delta:.10f}"
#                 )


#     def spend(self, epsilon_cost, delta_cost, mechanism_name=""):
#         """
#         Records a privacy spend and updates the spent budget.

#         Args:
#             epsilon_cost (float): Epsilon cost to spend.
#             delta_cost (float): Delta cost to spend.
#             mechanism_name (str): The name of the mechanism being used.
#         """
#         self._check_budget_expenditure(epsilon_cost, delta_cost)
        
#         self._spent_epsilon += epsilon_cost
#         self._spent_delta += delta_cost
#         self._spent_history.append((epsilon_cost, delta_cost, mechanism_name))

#         print(f"Budget spent: ({epsilon_cost:.4f}, {delta_cost:.10f}) for '{mechanism_name}'.")
#         print(f"Remaining budget: ({self.get_remaining_budget()[0]:.4f}, {self.get_remaining_budget()[1]:.10f})\n")

# # --- Modified DP Functions to use the BudgetAccountant ---
# # Assuming L1 sensitivity (L1_norm) for Laplace and L2 sensitivity (L2_norm) for Gaussian.
# # We'll use a fixed sensitivity of 1.0 for gradients as is common practice after clipping.
# # We will use this sensitivity to calculate the cost.

# def get_l1_sensitivity(tensor):
#     """Calculates the L1 sensitivity of a tensor."""
#     return torch.norm(tensor, p=1).item()

# def get_l2_sensitivity(tensor):
#     """Calculates the L2 sensitivity of a tensor."""
#     return torch.norm(tensor, p=2).item()

# def applyDPLaplace(tensor, epsilon, accountant=None, sensitivity=1.0):
#     """
#     Adds Laplace noise to a tensor to achieve epsilon-differential privacy.
#     Requires L1 sensitivity.

#     Args:
#         tensor (torch.Tensor or np.ndarray): The input data.
#         epsilon (float): The epsilon privacy parameter.
#         accountant (BudgetAccountant, optional): The budget accountant to track spend.
#         sensitivity (float): The L1 sensitivity of the query/function.
    
#     Returns:
#         torch.Tensor or np.ndarray: The tensor with added Laplace noise.
#     """
#     scale = sensitivity / epsilon
#     noise = np.random.laplace(loc=0, scale=scale, size=tensor.shape)
    
#     # Calculate privacy cost and spend it
#     if accountant:
#         # Laplace mechanism is (epsilon, 0)-DP
#         cost_epsilon, cost_delta = epsilon, 0.0
#         accountant.spend(cost_epsilon, cost_delta, mechanism_name="Laplace Noise")
    
#     if isinstance(tensor, torch.Tensor):
#         return tensor + torch.tensor(noise, dtype=tensor.dtype)
#     return tensor + noise

# def applyDPGaussian(tensor, epsilon, delta, accountant=None, sensitivity=1.0):
#     """
#     Adds Gaussian noise to a tensor to achieve (epsilon, delta)-differential privacy.
#     Requires L2 sensitivity.

#     Args:
#         tensor (torch.Tensor or np.ndarray): The input data.
#         epsilon (float): The epsilon privacy parameter.
#         delta (float): The delta privacy parameter.
#         accountant (BudgetAccountant, optional): The budget accountant to track spend.
#         sensitivity (float): The L2 sensitivity of the query/function.
    
#     Returns:
#         torch.Tensor or np.ndarray: The tensor with added Gaussian noise.
#     """
#     # Calculate noise scale (sigma) based on epsilon, delta, and sensitivity
#     # Formula: sigma >= (sensitivity * sqrt(2 * log(1.25 / delta))) / epsilon
#     # We'll use this to determine the cost if we are spending a specific amount of noise.
#     # To check the cost of a given noise scale, you'd solve for epsilon.
#     # Here, we assume epsilon/delta are given and we compute the required noise.
    
#     # A common formula for sigma is:
#     sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
#     noise = np.random.normal(loc=0, scale=sigma, size=tensor.shape)

#     # Calculate privacy cost and spend it
#     if accountant:
#         # Gaussian mechanism is (epsilon, delta)-DP
#         cost_epsilon, cost_delta = epsilon, delta
#         accountant.spend(cost_epsilon, cost_delta, mechanism_name="Gaussian Noise")
    
#     if isinstance(tensor, torch.Tensor):
#         return tensor + torch.tensor(noise, dtype=tensor.dtype)
#     return tensor + noise


# def applyClipping(tensor, max_norm):
#     """
#     Clips the L2 norm of the tensor to a maximum value.

#     Args:
#         tensor (torch.Tensor): The input tensor.
#         max_norm (float): The maximum allowed L2 norm.
    
#     Returns:
#         torch.Tensor: The clipped tensor.
#     """
#     norm = torch.norm(tensor, p=2)
#     if norm > max_norm:
#         tensor = tensor * (max_norm / norm)
#     return tensor

# def applyCountSketch(items, num_rows, num_cols):
#     """
#     Applies the Count Sketch algorithm to a list or tensor of items.

#     Args:
#         items (list, np.ndarray, or torch.Tensor): The data to sketch.
#         num_rows (int): Number of rows (hash functions) for the sketch.
#         num_cols (int): Number of columns (buckets) for the sketch.

#     Returns:
#         CSVec: The compressed Count Sketch object.
#     """
#     # Convert input to a flattened torch tensor
#     if isinstance(items, list):
#         items_tensor = torch.tensor(items, dtype=torch.float32)
#     elif isinstance(items, np.ndarray):
#         items_tensor = torch.from_numpy(items).float()
#     elif isinstance(items, torch.Tensor):
#         items_tensor = items.float()
#     else:
#         raise TypeError("Input items must be a list, numpy array, or torch tensor.")

#     dimension = items_tensor.numel()
    
#     # Create a CSVec (Count Sketch Vector) object
#     cs_vec = CSVec(d=dimension, c=num_cols, r=num_rows)
    
#     # Accumulate the vector into the sketch
#     cs_vec.accumulateVec(items_tensor)
    
#     return cs_vec

# def torch_to_list(tensor):
#     """Converts a torch tensor to a flattened Python list."""
#     return tensor.flatten().tolist()

# def list_to_numpy(data_list):
#     """Converts a list to a numpy array."""
#     return np.array(data_list)

# def getModelDimension(model):
#     """
#     Calculates the total number of parameters in a model.
#     """
#     total_params = 0
#     for param in model.parameters():
#         total_params += param.numel()
#     return total_params

# def add_noise_to_update(update_tensor, noise_multiplier):
#     """
#     Adds Gaussian noise to a tensor.
#     """
#     noise = torch.randn_like(update_tensor) * noise_multiplier
#     return update_tensor + noise

# # --- The update functions from your previous code, now with accountant support ---
# def client_update(client_model, optimizer, train_loader, epoch=5, use_privacy=False, privacy_method=None, clipping_norm=1.0, noise_multiplier=0.0, accountant=None):
#     """
#     Performs local training on a client's model for a number of epochs.
    
#     Args:
#         client_model: The local model.
#         optimizer: The optimizer for the local model.
#         train_loader: DataLoader for the local data.
#         epoch (int): Number of local epochs.
#         use_privacy (bool): Whether to apply privacy mechanisms.
#         privacy_method (str): 'laplace', 'gaussian', etc.
#         clipping_norm (float): The clipping bound for gradients.
#         noise_multiplier (float): Multiplier for Gaussian noise.
#         accountant (BudgetAccountant): The budget accountant to use.

#     Returns:
#         float: The final training loss.
#     """
#     client_model.train()
    
#     for e in range(epoch):
#         for batch_idx, (data, target) in enumerate(train_loader):
#             optimizer.zero_grad()
#             output = client_model(data)
#             loss = F.nll_loss(output, target)
#             loss.backward()

#             # --- Privacy and Clipping ---
#             if use_privacy:
#                 # Apply clipping first, as this bounds the sensitivity.
#                 # Clipping is essential for the privacy guarantees of DP-SGD.
#                 torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=clipping_norm)
                
#                 # Apply noise to each parameter's gradient
#                 if privacy_method == 'laplace' and accountant:
#                     for param in client_model.parameters():
#                         if param.grad is not None:
#                             # Use the clipped norm as the sensitivity
#                             sensitivity = clipping_norm
#                             applyDPLaplace(param.grad, noise_multiplier, accountant=accountant, sensitivity=sensitivity)
#                 elif privacy_method == 'gaussian' and accountant:
#                     # Note: You need epsilon, delta, and a sensitivity for this to work correctly.
#                     # The noise_multiplier is not a direct input for the cost calculation.
#                     # For a given noise_multiplier, the privacy cost can be calculated, but it's more complex.
#                     # A common approach is to set epsilon and delta and calculate the noise scale.
#                     # Let's assume noise_multiplier is related to sigma and we'll use a fixed epsilon/delta for the demo.
#                     # Here, we'll just use the `add_noise_to_update` which is more aligned with the original code.
#                     # To use the accountant, we'll need to define a cost for this operation.
                    
#                     # You need a delta for Gaussian, so we'll pass it in the main script.
#                     if accountant and accountant._total_delta <= 0:
#                         warnings.warn("Using Gaussian noise with a delta of 0 is not recommended for (epsilon,delta)-DP.")
                        
#                     for param in client_model.parameters():
#                         if param.grad is not None:
#                             # Add noise using the old method, and then account for the cost.
#                             param.grad.copy_(add_noise_to_update(param.grad, noise_multiplier))
#                             # Now, we need to spend the budget. The cost is derived from the clipping_norm and noise_multiplier.
#                             # For a (clipping_norm, sigma) pair, the cost is (epsilon, delta).
#                             # We'll calculate epsilon based on the sigma and clipping_norm.
#                             # epsilon = (clipping_norm / sigma) * sqrt(2 * log(1.25 / delta))
#                             # sigma is noise_multiplier here.
#                             if accountant:
#                                 # We need delta for this calculation. Let's assume it's passed with the accountant.
#                                 # Let's assume a default delta for this demo if not provided.
#                                 delta_for_cost = accountant.get_remaining_budget()[1] / (epoch * len(train_loader)) if accountant.get_remaining_budget()[1] > 0 else 1e-5
#                                 if clipping_norm > 0 and noise_multiplier > 0 and delta_for_cost > 0:
#                                     epsilon_cost = (clipping_norm / noise_multiplier) * np.sqrt(2 * np.log(1.25 / delta_for_cost))
#                                     accountant.spend(epsilon_cost, delta_for_cost, mechanism_name="Gaussian Noise (per batch)")
#                                 else:
#                                     warnings.warn("Could not calculate Gaussian privacy cost. Check noise_multiplier and clipping_norm.")
                    
#                 # We need to make sure the optimizer step uses the modified gradients
#                 optimizer.step()

#     return loss.item()

# # --- Example Usage ---

# # First, let's create a dummy model and data loader for the example.
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset

# # Dummy model
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.fc1 = nn.Linear(10, 5)
#         self.fc2 = nn.Linear(5, 2)
#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# # Dummy data
# dummy_data = torch.randn(100, 10)
# dummy_labels = torch.randint(0, 2, (100,))
# dummy_dataset = TensorDataset(dummy_data, dummy_labels)
# dummy_loader = DataLoader(dummy_dataset, batch_size=10)

# print("--- Example Usage of BudgetAccountant ---")

# # 1. Initialize the Budget Accountant
# # Let's set a total budget of epsilon=1.0 and delta=1e-5 for the entire training.
# total_epsilon = 1.0
# total_delta = 1e-5
# accountant = BudgetAccountant(total_epsilon=total_epsilon, total_delta=total_delta)

# print(f"Total budget initialized: Epsilon = {accountant._total_epsilon}, Delta = {accountant._total_delta}")
# print(f"Initial remaining budget: {accountant.get_remaining_budget()}\n")

# # 2. Use Laplace Noise and track the budget
# print("Applying Laplace Noise...")
# client_model_laplace = SimpleModel()
# optimizer_laplace = torch.optim.SGD(client_model_laplace.parameters(), lr=0.01)

# # We will run for 1 epoch, which has 10 batches (100 items / batch_size 10)
# # This will call applyDPLaplace 10 times in the client_update function.
# # Each call will spend a small part of the budget.
# # Let's set the noise_multiplier (which is epsilon here) per batch.
# # The `applyDPLaplace` function expects `epsilon` as a parameter.
# # We'll set a per-batch epsilon. Let's make it 0.05.
# # Total batches = 10 batches/epoch * 1 epoch = 10 batches.
# # Total expected epsilon spend = 10 * 0.05 = 0.5.
# # This should be within our budget of 1.0.

# _ = client_update(
#     client_model=client_model_laplace,
#     optimizer=optimizer_laplace,
#     train_loader=dummy_loader,
#     epoch=1,
#     use_privacy=True,
#     privacy_method='laplace',
#     clipping_norm=1.0, # Sensitivity will be clipped to 1.0
#     noise_multiplier=0.05, # This is the epsilon for the Laplace mechanism
#     accountant=accountant
# )

# print(f"\nTraining with Laplace noise complete.")
# print(f"Total spent budget so far: ({accountant.get_spent_budget()[0]:.4f}, {accountant.get_spent_budget()[1]:.10f})")
# print(f"Remaining budget: ({accountant.get_remaining_budget()[0]:.4f}, {accountant.get_remaining_budget()[1]:.10f})\n")

# # 3. Use Gaussian Noise and track the budget
# # Let's continue from where the budget is now.
# print("Applying Gaussian Noise...")
# client_model_gaussian = SimpleModel()
# optimizer_gaussian = torch.optim.SGD(client_model_gaussian.parameters(), lr=0.01)

# # We will run for 1 epoch, which has 10 batches.
# # Let's set the noise_multiplier (which is sigma here) to 0.5.
# # The code will calculate the epsilon cost based on this noise.
# # We'll assume the budget's remaining delta is split across batches.
# remaining_delta = accountant.get_remaining_budget()[1]
# num_batches = len(dummy_loader) * 1 # 1 epoch
# delta_per_batch = remaining_delta / num_batches
# print(f"Delta to spend per batch: {delta_per_batch:.10f}")

# _ = client_update(
#     client_model=client_model_gaussian,
#     optimizer=optimizer_gaussian,
#     train_loader=dummy_loader,
#     epoch=1,
#     use_privacy=True,
#     privacy_method='gaussian',
#     clipping_norm=1.0, # Sensitivity will be clipped to 1.0
#     noise_multiplier=0.5, # This is sigma for Gaussian
#     accountant=accountant # Pass the same accountant
# )

# print(f"\nTraining with Gaussian noise complete.")
# print(f"Final remaining budget: ({accountant.get_remaining_budget()[0]:.4f}, {accountant.get_remaining_budget()[1]:.10f})")
# print(f"Total budget spent: ({accountant.get_spent_budget()[0]:.4f}, {accountant.get_spent_budget()[1]:.10f})")
# print("\nSpending History:")
# for spend in accountant.get_spent_history():
#     print(f"  - Mechanism: {spend[2]}, Epsilon Cost: {spend[0]:.4f}, Delta Cost: {spend[1]:.10f}")


# # 4. Example of Exceeding the Budget
# print("\n--- Example of Exceeding the Budget ---")
# # Reset the accountant with a very small budget
# tiny_accountant = BudgetAccountant(total_epsilon=0.1, total_delta=1e-8)
# print(f"New tiny budget: Epsilon = {tiny_accountant._total_epsilon}, Delta = {tiny_accountant._total_delta}")

# try:
#     _ = client_update(
#         client_model=SimpleModel(),
#         optimizer=torch.optim.SGD(SimpleModel().parameters(), lr=0.01),
#         train_loader=dummy_loader,
#         epoch=1,
#         use_privacy=True,
#         privacy_method='laplace',
#         clipping_norm=1.0,
#         noise_multiplier=0.05, # Cost per batch is 0.05. We have 10 batches. Total is 0.5 > 0.1.
#         accountant=tiny_accountant
#     )
# except RuntimeError as e:
#     print(f"\nSuccessfully caught budget exhaustion error:")
#     print(e)
#     print(f"Budget remaining when error occurred: {tiny_accountant.get_remaining_budget()}")



import torch
import torch.nn.functional as F
import numpy as np
import warnings
from csvec.csvec import CSVec

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