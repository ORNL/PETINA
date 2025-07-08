import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from torch.utils.data import DataLoader, TensorDataset
from PETINA import BudgetAccountant, Budget, BudgetError,Sketching,CSVec

# -------------------------------
# Data Conversion Helper Functions (Keep these as they are)
# -------------------------------

def numpy_to_list(nd_array):
    """Converts a NumPy array to a flattened list and returns its original shape."""
    flattened_list = nd_array.flatten().tolist()
    nd_array_shape = nd_array.shape
    return flattened_list, nd_array_shape


def list_to_numpy(flattened_list, nd_array_shape):
    """Converts a flattened list back to a NumPy array with the given shape."""
    reverted_ndarray = np.array(flattened_list).reshape(nd_array_shape)
    return reverted_ndarray


def torch_to_list(torch_tensor):
    """Converts a PyTorch tensor to a flattened list and returns its original shape."""
    flattened_list = torch_tensor.flatten().tolist()
    tensor_shape = torch_tensor.shape
    return flattened_list, tensor_shape


def list_to_torch(flattened_list, tensor_shape):
    """Converts a flattened list back to a PyTorch tensor with the given shape."""
    reverted_tensor = torch.as_tensor(flattened_list).reshape(tensor_shape)
    return reverted_tensor


def type_checking_and_return_lists(domain):
    """
    Converts the input data (tensor, numpy array, or list) to a list and returns its shape (if applicable).
    """
    if isinstance(domain, torch.Tensor):
        items, shape = torch_to_list(domain)
    elif isinstance(domain, np.ndarray):
        items, shape = numpy_to_list(domain)
    elif isinstance(domain, list):
        items = domain
        shape = 0
    else:
        raise ValueError("only takes list, ndarray, tensor type")
    return items, shape


def type_checking_return_actual_dtype(domain, result, shape):
    """
    Converts a processed list back to the original data type of 'domain'.
    """
    if isinstance(domain, torch.Tensor):
        return list_to_torch(result, shape)
    elif isinstance(domain, np.ndarray):
        return list_to_numpy(result, shape)
    else:
        return result

# -------------------------------
# Differential Privacy Mechanisms (Modified to use BudgetAccountant)
# -------------------------------

def applyDPLaplace(domain, sensitivity=1, epsilon=0.01, gamma=1, accountant=None):
    """
    Applies Laplace noise to the input data for differential privacy.
    Modified to track budget with a BudgetAccountant.

    Parameters:
        domain: Input data (list, numpy array, or tensor).
        sensitivity: Maximum change by a single individual's data (default: 1).
        epsilon: Privacy parameter (default: 1).
        gamma: Scaling factor for noise (default: 1).
        accountant (BudgetAccountant, optional): The budget accountant to track spend.

    Returns:
        Data with added Laplace noise in the same format as the input.
    """
    # Use helper functions to convert input data to a flattened list and get its shape
    data, shape = type_checking_and_return_lists(domain)
    
    # Calculate the scale for the Laplace distribution.
    # This maintains the original noise calculation from the PETINA function.
    scale = sensitivity * gamma / epsilon
    
    # Add Laplace noise to each element of the flattened data.
    privatized = np.array(data) + np.random.laplace(loc=0, scale=scale, size=len(data))

    # --- Inject the budget tracking logic here ---
    if accountant is not None:
        print("Accountant is present, spending budget for Laplace noise addition.")
        
        # The budget cost is based on the `epsilon` parameter provided to the function.
        # For Laplace noise, the delta cost is 0.
        cost_epsilon, cost_delta = epsilon, 0.0
        
        # The `spend` method will internally check if the budget is exceeded.
        accountant.spend(cost_epsilon, cost_delta)
        
        # Print the total spent budget for debugging and monitoring purposes.
        print("Total spend: %r" % (accountant.total(),))
        
    # Convert the processed flattened list back to the original data type and shape.
    return type_checking_return_actual_dtype(domain, privatized, shape)

def applyDPGaussian(domain, delta=1e-5, epsilon=1.0, gamma=1.0, accountant=None):
    """
    Applies Gaussian noise to the input data for differential privacy,
    and optionally tracks budget via a BudgetAccountant.

    Parameters:
        domain: Input data (list, numpy array, or tensor).
        delta (float): Failure probability (default: 1e-5).
        epsilon (float): Privacy parameter (default: 1.0).
        gamma (float): Scaling factor for noise (default: 1.0).
        accountant (BudgetAccountant, optional): Tracks spend for (ε, δ).

    Returns:
        Data with added Gaussian noise in the same format as the input.
    """
    # Flatten to list
    data, shape = type_checking_and_return_lists(domain)

    # Compute σ for (ε, δ)-Gaussian mechanism
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * gamma / epsilon

    # Add Gaussian noise
    privatized = np.array(data) + np.random.normal(loc=0, scale=sigma, size=len(data))

    # Budget accounting
    if accountant is not None:
        # Spend the exact (ε, δ) for this invocation
        accountant.spend(epsilon, delta)
        # (Optional) debug print
        print(f"Gaussian: spent ε={epsilon}, δ={delta}; remaining={accountant.remaining()}")

    # Restore to original type/shape
    return type_checking_return_actual_dtype(domain, privatized, shape)


# def applyClipping(tensor, max_norm):
#     """
#     Clips the L2 norm of the tensor to a maximum value.
#     This function is unchanged, as clipping is a pre-processing step, not a DP mechanism that consumes budget directly.
#     """
#     norm = torch.norm(tensor, p=2)
#     if norm > max_norm:
#         tensor = tensor * (max_norm / norm)
#     return tensor

# def applyCountSketch(items, num_rows, num_cols):
#     """
#     Applies the Count Sketch algorithm to a list or tensor of items.
#     This function returns the CSVec object. Unsketching happens at the server side.
#     This function is not directly privacy-consuming in the way Laplace/Gaussian are,
#     but it enables communication efficiency and randomization for DP.
#     """
#     if isinstance(items, list):
#         items_tensor = torch.tensor(items, dtype=torch.float32)
#     elif isinstance(items, np.ndarray):
#         items_tensor = torch.from_numpy(items).float()
#     elif isinstance(items, torch.Tensor):
#         items_tensor = items.float()
#     else:
#         raise TypeError("Input items must be a list, numpy array, or torch tensor.")

#     dimension = items_tensor.numel()
    
#     cs_vec = CSVec(d=dimension, c=num_cols, r=num_rows)
#     cs_vec.accumulateVec(items_tensor)
    
#     return cs_vec

# Helper function to get model dimension for CSVec
def getModelDimension(model):
    """
    Calculates the total number of parameters in a model.
    """
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params

def getModelParameterFlatten(model):
    """
    Given a model, get the parameter tensors as 1D tensors.
    """
    params = [p.detach().view(-1) for p in model.parameters()]
    flat_tensor = torch.cat(params)
    return flat_tensor

def add_noise_to_update(update, noise_multiplier):
    """ Adds Gaussian noise to the update for differential privacy. """
    noise = torch.randn_like(update) * noise_multiplier
    return update + noise

def tensor_to_parameters(model, flat_tensor):
    """
    Converts a flat tensor back into model parameters.
    """
    offset = 0
    new_params = []
    for param in model.parameters():
        param_size = param.numel()
        new_param = flat_tensor[offset:offset + param_size].view(param.size())
        new_params.append(new_param)
        offset += param_size
    return new_params

def client_update(client_model, optimizer, train_loader, epoch=5, use_privacy=False, privacy_method=None, clipping_norm=1.0, noise_multiplier=0.0, accountant=None):
    """
    Performs local training on a client's model, applies DP mechanisms, and tracks budget.
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
                # 1. Apply clipping first to bound the sensitivity. Essential for DP-SGD.
                torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=clipping_norm)
                
                # 2. Apply noise and track budget with a try-except block
                try:
                    # Calculate per-parameter, per-batch budget.
                    # We need the total number of parameters for the Gaussian budget split.
                    total_params = getModelDimension(client_model)
                    num_batches_per_epoch = len(train_loader)
                    
                    if privacy_method == 'laplace':
                        # The `noise_multiplier` is passed as `epsilon_per_param` to applyDPLaplace.
                        for param in client_model.parameters():
                            if param.grad is not None:
                                applyDPLaplace(param.grad,sensitivity=clipping_norm, accountant=accountant)
                                
                    elif privacy_method == 'gaussian':
                        # Distribute delta budget over all parameters and all batches in the client update.
                        # This assumes non-adaptive composition within the client update.
                        # The total delta from the accountant is assumed to be shared across all such calls.
                        # We are passing the `epsilon_per_param` and `delta_per_param` to `applyDPGaussian`.
                        
                        # This specific calculation for delta_per_param requires careful thought about overall budget.
                        # A simpler way often is to have a fixed epsilon/delta for EACH gradient addition.
                        # For this example, we will calculate delta per parameter per batch.
                        
                        # Total gradient additions = epochs * num_batches_per_epoch * total_params
                        total_gradient_ops = epoch * num_batches_per_epoch * total_params
                        
                        # Use a small default delta if the remaining budget's delta is 0 or too small.
                        # This prevents division by zero or extremely large epsilon costs.
                        remaining_budget = accountant.remaining()
                        if remaining_budget.delta > 0:
                            delta_for_gaussian_cost = remaining_budget.delta / total_gradient_ops
                        else:
                            # Fallback if remaining delta is zero, just to allow calculation to proceed for demo.
                            # In a real scenario, this means no more delta budget.
                            delta_for_gaussian_cost = 1e-10 # A tiny non-zero delta

                        if delta_for_gaussian_cost <= 0:
                            raise BudgetError("Calculated delta per parameter is zero or negative, cannot apply Gaussian noise.")
                        
                        # `noise_multiplier` is `sigma` for Gaussian here.
                        sigma = noise_multiplier
                        
                        # Calculate epsilon for this (sigma, sensitivity, delta_per_param)
                        # epsilon = (sensitivity * sqrt(2 * log(1.25 / delta))) / sigma
                        epsilon_for_gaussian_cost = (clipping_norm * np.sqrt(2 * np.log(1.25 / delta_for_gaussian_cost))) / sigma
                        
                        for param in client_model.parameters():
                            if param.grad is not None:
                                # Apply the noise. Note: noise_multiplier is sigma.
                                param.grad.copy_(add_noise_to_update(param.grad, noise_multiplier))
                                # Now spend the calculated cost for this specific parameter's noise addition.
                                accountant.spend(epsilon_for_gaussian_cost, delta_for_gaussian_cost)

                except BudgetError as e:
                    print(f"\n--- Budget Exhausted! Stopping local training gracefully. ---")
                    print(f"Reason: {e}")
                    # Return a value that indicates early stopping, or re-raise if you want it handled higher up
                    return loss.item() # Exit the function early
                except Exception as e:
                    # Catch any other unexpected errors in the DP block
                    print(f"\n--- An unexpected error occurred during DP application! ---")
                    print(f"Error: {e}")
                    return loss.item() # Exit the function early

            optimizer.step()

    return loss.item()


def server_aggregate_models(global_model, client_models):
    """Aggregates models by averaging their state dictionaries."""
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k] for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

def server_aggregate_sketches(global_model, client_diff_sketches, sketch_column, sketch_row, num_selected):
    """Aggregates sketches from clients."""
    parameter_dimension = getModelDimension(global_model)
    summed_diff_Sketch = CSVec(parameter_dimension, int(sketch_column), int(sketch_row))
    summed_diff_Sketch.zero() # Ensure it starts at zero

    for i in range(num_selected):
        summed_diff_Sketch += client_diff_sketches[i]
    
    # Normalize the sketch if needed
    summed_diff_Sketch = summed_diff_Sketch / (num_selected) 
    return summed_diff_Sketch

def update_parameters_in_clients(summed_diff_sketch, global_model, client_models, num_selected):
    """Unsketches aggregated updates and applies them to global and client models."""
    unsketched_flattened_tensor = summed_diff_sketch.unSketch(k=getModelDimension(client_models[0]))
    
    reconstructed_params_for_global = tensor_to_parameters(global_model, unsketched_flattened_tensor)
    
    # Apply these to the global model.
    with torch.no_grad(): # Ensure no gradient tracking for direct parameter manipulation
        for global_param, recon_param in zip(global_model.parameters(), reconstructed_params_for_global):
            global_param.data += recon_param # Apply the aggregated update

    # Now, update client models to match the new global model
    for client_model in client_models:
        client_model.load_state_dict(global_model.state_dict())

def test(global_model, test_loader):
    """Evaluates the model on the test set."""
    global_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = global_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    return test_loss, acc

# --- Dummy Model and Data from the User's Request ---
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

def init_weights(m):
    """Initializes model weights."""
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# -------------------------------
# Core Federated Learning Loop
# -------------------------------

def run_federated_learning(
    sketch_column, sketch_row, num_selected, total_epsilon, total_delta,
    method, num_clients, num_rounds, epochs, batch_size,
    clipping_norm, noise_multiplier
):
    """
    Runs a federated learning simulation with privacy mechanisms and budget tracking.
    """
    global_model = SimpleModel()
    global_model.apply(init_weights)
    client_models = [SimpleModel() for _ in range(num_clients)]
    client_old_models = [SimpleModel() for _ in range(num_clients)]

    for model in client_models:
        model.apply(init_weights)
        model.load_state_dict(global_model.state_dict())

    opt = [torch.optim.SGD(model.parameters(), lr=0.1) for model in client_models]

    # --- Use dummy data and split it among clients ---
    dummy_data = torch.randn(1000, 10) # Larger dataset for more batches
    dummy_labels = torch.randint(0, 2, (1000,))
    dummy_dataset = TensorDataset(dummy_data, dummy_labels)
    
    # --- FIXED: Use a more robust split logic to handle remainders ---
    dataset_size = len(dummy_dataset)
    base_size = dataset_size // num_clients
    remainder = dataset_size % num_clients
    
    # Create the list of lengths for each split, distributing the remainder
    lengths = [base_size + 1] * remainder + [base_size] * (num_clients - remainder)
    
    # Sanity check to ensure the sum of lengths equals the dataset size
    assert sum(lengths) == dataset_size, "Sum of split lengths does not match dataset size!"
    
    traindata_split = torch.utils.data.random_split(dummy_dataset, lengths)
    train_loader = [DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]

    # Dummy test data loader
    dummy_test_data = torch.randn(200, 10)
    dummy_test_labels = torch.randint(0, 2, (200,))
    dummy_test_dataset = TensorDataset(dummy_test_data, dummy_test_labels)
    test_loader = DataLoader(dummy_test_dataset, batch_size=batch_size, shuffle=False)

    test_loss_all = []
    acc_all = []
    train_loss_all = []

    # Initialize the Budget Accountant for the entire FL simulation
    accountant = BudgetAccountant(epsilon=total_epsilon, delta=total_delta)
    print(f"\n--- FL Simulation with Budget Accountancy ---")
    print(f"Total budget initialized: Epsilon = {accountant.epsilon}, Delta = {accountant.delta}")
    print(f"Initial remaining budget: {accountant.remaining()}\n")

    for r in range(num_rounds):
        print(f"--- Round {r+1}/{num_rounds} ---")
        client_idx = np.random.permutation(num_clients)[:num_selected]
        
        # Save old params if using count_sketch for diff calculation
        if method == 'count_sketch':
            for i in range(num_selected):
                client_old_models[i].load_state_dict(client_models[i].state_dict())
                
        # --- Client Update Phase ---
        round_losses = []
        for i in range(num_selected):
            # Pass the accountant to each client update
            loss = client_update(
                client_model=client_models[i],
                optimizer=opt[i],
                train_loader=train_loader[client_idx[i]],
                epoch=epochs,
                use_privacy=True, # Always use privacy for this demo
                privacy_method=method,
                clipping_norm=clipping_norm,
                noise_multiplier=noise_multiplier,
                accountant=accountant
            )
            round_losses.append(loss)
            
            # Check if budget was exhausted during this client's update
            if loss is None: # client_update returns None on BudgetError
                print(f"Budget exhausted during client {i+1} update. Stopping FL simulation.")
                return max(acc_all) if acc_all else 0.0 # Return current max accuracy or 0

        # --- Server Aggregation Phase ---
        if method == 'count_sketch':
            client_diff_sketches = []
            for i in range(num_selected):
                # Calculate diff and sketch it
                diff_tensor = getModelParameterFlatten(client_models[i]) - getModelParameterFlatten(client_old_models[i])
                sketched_diff = Sketching.applyCountSketch(diff_tensor, sketch_row, sketch_column)
                client_diff_sketches.append(sketched_diff)
            
            summed_diff_Sketch = server_aggregate_sketches(global_model, client_diff_sketches, sketch_column, sketch_row, num_selected)
            update_parameters_in_clients(summed_diff_Sketch, global_model, client_models, num_selected)
        else:
            # For Laplace/Gaussian (which directly modify client models in client_update)
            # The server simply aggregates the (noisy) models.
            server_aggregate_models(global_model, client_models)
            
        test_loss, acc = test(global_model, test_loader)
        train_loss_all.append(np.mean(round_losses))
        test_loss_all.append(test_loss)
        acc_all.append(acc)

        print(f"Round {r+1}: Avg Train Loss = {np.mean(round_losses):.3g} | Test Loss = {test_loss:.3g} | Test Acc = {acc:.3f}")
        print(f"Remaining budget: {accountant.remaining()}\n")

        # After each round, check if the budget is exhausted to avoid unnecessary computations
        eps, delt = accountant.remaining()
        if eps <= 0 and delt <= 0:
            print(f"Budget fully exhausted after round {r+1}. Stopping FL simulation.")
            break # Exit the main FL loop

    print(f"\n--- FL Simulation Finished ---")
    print(f"Final spent budget: {accountant.total()}")
    print(f"Total budget: ({accountant.epsilon}, {accountant.delta})")
    print(f"Max accuracy achieved: {max(acc_all) if acc_all else 0:.4f}")
    
    return max(acc_all) if acc_all else 0.0


if __name__ == "__main__":
    # --- Example Usage of Graceful Budget Catching ---
    print("--- Example Usage of Graceful Budget Catching ---")

    # Define FL and DP parameters for the simulation
    sketch_row = 16 # For count sketch
    sketch_column = 1024 # For count sketch

    num_clients = 30
    num_selected = 10 # Number of clients selected per round
    num_rounds = 5 # Number of FL rounds
    epochs = 1 # Local epochs per client
    batch_size = 10
    clipping_norm = 1.0 # Gradient clipping norm (sensitivity)
    
    # Differential Privacy parameters for the entire simulation (total budget)
    total_epsilon_budget = 2
    total_delta_budget = 1e-5 # Typical small delta for Gaussian DP

    # Noise multiplier for the DP mechanisms (this is related to the per-step epsilon/sigma)
    # For Laplace: this is the epsilon per parameter.
    # For Gaussian: this is `sigma` for the noise distribution.
    noise_param_val = 0.5 # Example value; adjust based on desired privacy/utility trade-off

    # --- Run with Laplace Noise ---
    print("\n\n=============== Running with Laplace Noise ===============")
    run_federated_learning(
        sketch_column=sketch_column,
        sketch_row=sketch_row,
        num_selected=num_selected,
        total_epsilon=total_epsilon_budget,
        total_delta=0.0, # Laplace is (epsilon, 0)-DP
        method='laplace',
        num_clients=num_clients,
        num_rounds=num_rounds,
        epochs=epochs,
        batch_size=batch_size,
        clipping_norm=clipping_norm,
        noise_multiplier=0.01 # Epsilon per parameter in Laplace
    )

    # # --- Run with Gaussian Noise ---
    # print("\n\n=============== Running with Gaussian Noise ===============")
    # run_federated_learning(
    #     sketch_column=sketch_column,
    #     sketch_row=sketch_row,
    #     num_selected=num_selected,
    #     total_epsilon=total_epsilon_budget,
    #     total_delta=total_delta_budget,
    #     method='gaussian',
    #     num_clients=num_clients,
    #     num_rounds=num_rounds,
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     clipping_norm=clipping_norm,
    #     noise_multiplier=0.5 # Sigma for Gaussian noise
    # )

    # --- Run with Count Sketch (as communication efficiency) ---
    # print("\n\n=============== Running with Count Sketch (as communication efficiency) ===============")
    # run_federated_learning(
    #     sketch_column=sketch_column,
    #     sketch_row=sketch_row,
    #     num_selected=num_selected,
    #     total_epsilon=total_epsilon_budget,
    #     total_delta=total_delta_budget,
    #     method='count_sketch',
    #     num_clients=num_clients,
    #     num_rounds=num_rounds,
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     clipping_norm=clipping_norm,
    #     noise_multiplier=0.0 # Not directly used by count_sketch as a noise param
    # )

    # --- Example of Budget Exhaustion ---
    # print("\n\n=============== Demonstrating Budget Exhaustion ===============")
    # # Set a very tight budget for a quick exhaustion example
    # tiny_epsilon = 0.001 # Very small epsilon
    # tiny_delta = 1e-15 # Very small delta
    
    # run_federated_learning(
    #     sketch_column=sketch_column,
    #     sketch_row=sketch_row,
    #     num_selected=num_selected,
    #     total_epsilon=tiny_epsilon,
    #     total_delta=tiny_delta,
    #     method='laplace', # Choose a method that spends budget
    #     num_clients=num_clients,
    #     num_rounds=5, # Try to run for a few rounds
    #     epochs=1,
    #     batch_size=batch_size,
    #     clipping_norm=clipping_norm,
    #     noise_multiplier=0.001 # This will cause quick exhaustion
    # )