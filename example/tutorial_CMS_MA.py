"""
This script is based on the original example.py from the pure-LDP library, but has been adapted to
include the PETINA CMS implementation and a centralized CMS implementation. It demonstrates how to
use both the PETINA CMS and a centralized CMS for frequency estimation under Local Differential Privacy (LDP).
It also includes a simulation of heavy hitters using the Prefix Extending Method (PEM).

This script simulates frequency estimation using various pure Local Differential Privacy (LDP) mechanisms
on synthetic categorical data. It:
  1. Generates synthetic data uniformly over a finite domain
  2. Applies several frequency oracle algorithms under pure-LDP and PETINA CMS
  3. Estimates the data distribution
  4. Evaluates and visualizes the estimation error

Algorithms used:
  - Optimized Unary Encoding (OUE)
  - Direct Encoding (DE)
  - RAPPOR
  - Hadamard Response (HR)

Source: https://github.com/Samuel-Maddock/pure-LDP
"""
from pure_ldp.frequency_oracles import *
from pure_ldp.heavy_hitters import *
import numpy as np
from collections import Counter
import random
import math
from scipy.linalg import hadamard # Make sure this is imported if you use Hadamard
import warnings
from numbers import Integral, Real

# ---- Import from PETINA ----
from PETINA import generate_hash_funcs, Client_PETINA_CMS, Server_PETINA_CMS, centralized_count_mean_sketch

# --- Start of BudgetAccountant and utility classes source code ---

# These classes are from the diffprivlib library and are included here for a self-contained script.

class Budget(tuple):
    """Custom tuple subclass for privacy budgets of the form (epsilon, delta)."""
    def __new__(cls, epsilon, delta):
        if epsilon < 0:
            raise ValueError("Epsilon must be non-negative")
        if not 0 <= delta <= 1:
            raise ValueError("Delta must be in [0, 1]")
        return tuple.__new__(cls, (epsilon, delta))

    def __gt__(self, other):
        if self.__ge__(other) and not self.__eq__(other): return True
        return False
    def __ge__(self, other):
        if self[0] >= other[0] and self[1] >= other[1]: return True
        return False
    def __lt__(self, other):
        if self.__le__(other) and not self.__eq__(other): return True
        return False
    def __le__(self, other):
        if self[0] <= other[0] and self[1] <= other[1]: return True
        return False
    def __repr__(self):
        return f"(epsilon={self[0]}, delta={self[1]})"

class BudgetError(ValueError):
    """Custom exception to capture the privacy budget being exceeded."""

class PrivacyLeakWarning(RuntimeWarning):
    """Custom warning to capture privacy leaks."""

class DiffprivlibCompatibilityWarning(RuntimeWarning):
    """Custom warning to capture inherited class arguments that are not compatible with diffprivlib."""

warnings.simplefilter('always', PrivacyLeakWarning)

def check_epsilon_delta(epsilon, delta, allow_zero=False):
    """Checks that epsilon and delta are valid values for differential privacy."""
    if not isinstance(epsilon, Real) or not isinstance(delta, Real):
        raise TypeError("Epsilon and delta must be numeric")
    if epsilon < 0:
        raise ValueError("Epsilon must be non-negative")
    if not 0 <= delta <= 1:
        raise ValueError("Delta must be in [0, 1]")
    if not allow_zero and epsilon + delta == 0:
        raise ValueError("Epsilon and Delta cannot both be zero")

class BudgetAccountant:
    """Privacy budget accountant for differential privacy."""
    _default = None
    def __init__(self, epsilon=float("inf"), delta=1.0, slack=0.0, spent_budget=None):
        check_epsilon_delta(epsilon, delta)
        self.__epsilon = epsilon
        self.__min_epsilon = 0 if epsilon == float("inf") else epsilon * 1e-14
        self.__delta = delta
        self.__spent_budget = []
        self.slack = slack
        if spent_budget is not None:
            if not isinstance(spent_budget, list): raise TypeError("spent_budget must be a list")
            for _epsilon, _delta in spent_budget: self.spend(_epsilon, _delta)
    def __repr__(self, n_budget_max=5):
        params = []
        if self.epsilon != float("inf"): params.append(f"epsilon={self.epsilon}")
        if self.delta != 1: params.append(f"delta={self.delta}")
        if self.slack > 0: params.append(f"slack={self.slack}")
        if self.spent_budget:
            if len(self.spent_budget) > n_budget_max: params.append("spent_budget=" + str(self.spent_budget[:n_budget_max] + ["..."]).replace("'", ""))
            else: params.append("spent_budget=" + str(self.spent_budget))
        return "BudgetAccountant(" + ", ".join(params) + ")"
    def __enter__(self):
        self.old_default = self.pop_default()
        self.set_default()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pop_default()
        if self.old_default is not None: self.old_default.set_default()
        del self.old_default
    def __len__(self): return len(self.spent_budget)
    @property
    def slack(self): return self.__slack
    @slack.setter
    def slack(self, slack):
        if not 0 <= slack <= self.delta: raise ValueError(f"Slack must be between 0 and delta ({self.delta}), inclusive. Got {slack}.")
        epsilon_spent, delta_spent = self.total(slack=slack)
        if self.epsilon < epsilon_spent or self.delta < delta_spent: raise BudgetError(f"Privacy budget will be exceeded by changing slack to {slack}.")
        self.__slack = slack
    @property
    def spent_budget(self): return self.__spent_budget.copy()
    @property
    def epsilon(self): return self.__epsilon
    @property
    def delta(self): return self.__delta
    def total(self, spent_budget=None, slack=None):
        if spent_budget is None: spent_budget = self.spent_budget
        else:
            for epsilon, delta in spent_budget: check_epsilon_delta(epsilon, delta)
        if slack is None: slack = self.slack
        elif not 0 <= slack <= self.delta: raise ValueError(f"Slack must be between 0 and delta ({self.delta}), inclusive. Got {slack}.")
        epsilon_sum, epsilon_exp_sum, epsilon_sq_sum = 0, 0, 0
        for epsilon, _ in spent_budget:
            epsilon_sum += epsilon
            epsilon_exp_sum += (1 - np.exp(-epsilon)) * epsilon / (1 + np.exp(-epsilon))
            epsilon_sq_sum += epsilon ** 2
        total_epsilon_naive = epsilon_sum
        total_delta = self.__total_delta_safe(spent_budget, slack)
        if slack == 0: return Budget(total_epsilon_naive, total_delta)
        total_epsilon_drv = epsilon_exp_sum + np.sqrt(2 * epsilon_sq_sum * np.log(1 / slack))
        total_epsilon_kov = epsilon_exp_sum + np.sqrt(2 * epsilon_sq_sum * np.log(np.exp(1) + np.sqrt(epsilon_sq_sum) / slack))
        return Budget(min(total_epsilon_naive, total_epsilon_drv, total_epsilon_kov), total_delta)
    def check(self, epsilon, delta):
        check_epsilon_delta(epsilon, delta)
        if self.epsilon == float("inf") and self.delta == 1: return True
        if 0 < epsilon < self.__min_epsilon: raise ValueError(f"Epsilon must be at least {self.__min_epsilon} if non-zero, got {epsilon}.")
        spent_budget = self.spent_budget + [(epsilon, delta)]
        if Budget(self.epsilon, self.delta) >= self.total(spent_budget=spent_budget): return True
        raise BudgetError(f"Privacy spend of ({epsilon},{delta}) not permissible; will exceed remaining privacy budget."
                          f" Use {self.__class__.__name__}.{self.remaining.__name__}() to check remaining budget.")
    def remaining(self, k=1):
        if not isinstance(k, Integral): raise TypeError(f"k must be integer-valued, got {type(k)}.")
        if k < 1: raise ValueError(f"k must be at least 1, got {k}.")
        _, spent_delta = self.total()
        delta = 1 - ((1 - self.delta) / (1 - spent_delta)) ** (1 / k) if spent_delta < 1.0 else 1.0
        lower = 0; upper = self.epsilon; old_interval_size = (upper - lower) * 2
        while old_interval_size > upper - lower:
            old_interval_size = upper - lower; mid = (upper + lower) / 2
            spent_budget = self.spent_budget + [(mid, 0)] * k
            x_0, _ = self.total(spent_budget=spent_budget)
            if x_0 >= self.epsilon: upper = mid
            if x_0 <= self.epsilon: lower = mid
        epsilon = (upper + lower) / 2; return Budget(epsilon, delta)
    def spend(self, epsilon, delta):
        self.check(epsilon, delta); self.__spent_budget.append((epsilon, delta)); return self
    @staticmethod
    def __total_delta_safe(spent_budget, slack):
        delta_spend = [slack]
        for _, delta in spent_budget: delta_spend.append(delta)
        delta_spend.sort()
        prod = 0
        for delta in delta_spend: prod += delta - prod * delta
        return prod
    @staticmethod
    def load_default(accountant):
        if accountant is None:
            if BudgetAccountant._default is None: BudgetAccountant._default = BudgetAccountant()
            return BudgetAccountant._default
        if not isinstance(accountant, BudgetAccountant): raise TypeError(f"Accountant must be of type BudgetAccountant, got {type(accountant)}")
        return accountant
    def set_default(self): BudgetAccountant._default = self; return self
    @staticmethod
    def pop_default(): default = BudgetAccountant._default; BudgetAccountant._default = None; return default
# --- End of BudgetAccountant and utility classes source code ---

# Super simple synthetic dataset
data = np.concatenate(
    (
        [1] * 8000,
        [2] * 4000,
        [3] * 1000,
        [4] * 500,
        [5] * 1000,
        [6] * 1800,
        [7] * 2000,
        [8] * 300,
    )
)
original_freq = list(Counter(data).values())  # True frequencies of the dataset

# Parameters for experiment
epsilon = 3
d = 8
is_the = True
is_oue = True
is_olh = True

# --- Initialize a single BudgetAccountant for the entire experiment ---
# This accountant tracks the total privacy budget spent by the server as it aggregates client contributions.
# We set a ceiling of infinity to track the total budget without stopping the simulation.
accountant = BudgetAccountant(epsilon=float("inf"), delta=1.0)

print(f"--- Initializing a single BudgetAccountant for the experiment ---")
print(f"Starting total budget spent: {accountant.total()}")

# Optimal Local Hashing (OLH)
client_olh = LHClient(epsilon=epsilon, d=d, use_olh=is_olh)
server_olh = LHServer(epsilon=epsilon, d=d, use_olh=is_olh)

# Optimal Unary Encoding (OUE)
client_oue = UEClient(epsilon=epsilon, d=d, use_oue=is_oue)
server_oue = UEServer(epsilon=epsilon, d=d, use_oue=is_oue)

# Threshold Histogram Encoding (THE)
client_the = HEClient(epsilon=epsilon, d=d)
server_the = HEServer(epsilon=epsilon, d=d, use_the=is_the)

# Hadamard Response (HR)
server_hr = HadamardResponseServer(epsilon, d)
client_hr = HadamardResponseClient(epsilon, d, server_hr.get_hash_funcs())

# Apple's Count Mean Sketch (CMS)
k = 128 # 128 hash functions
m = 1024 # Each hash function maps to the domain {0, ... 1023}

server_cms = CMSServer(epsilon, k, m)
client_cms = CMSClient(epsilon, server_cms.get_hash_funcs(), m)

# --- Simulate client-side privatisation + server-side aggregation for LDP methods ---
# We spend the budget for each item as it is aggregated by the server.
print("\n--- Simulating Aggregation for Pure-LDP Methods (OLH, OUE, THE, HR) ---")
for i, item in enumerate(data):
    priv_olh_data = client_olh.privatise(item)
    priv_oue_data = client_oue.privatise(item)
    priv_the_data = client_the.privatise(item)
    priv_hr_data = client_hr.privatise(item)

    server_olh.aggregate(priv_olh_data)
    server_oue.aggregate(priv_oue_data)
    server_the.aggregate(priv_the_data)
    server_hr.aggregate(priv_hr_data)

    # Record the privacy budget spent for this single client contribution.
    # We use the configured epsilon and delta=0 as these are pure-LDP mechanisms.
    accountant.spend(epsilon=epsilon, delta=0.0)

    # Print a progress update every 5000 items
    if (i + 1) % 5000 == 0:
        print(f"  Processed {i+1} clients. Total budget spent: {accountant.total()}")

# Simulate server-side estimation for LDP methods
oue_estimates = []
olh_estimates = []
the_estimates = []
hr_estimates = []

for i in range(0, d):
    olh_estimates.append(round(server_olh.estimate(i + 1)))
    oue_estimates.append(round(server_oue.estimate(i + 1)))
    the_estimates.append(round(server_the.estimate(i + 1)))
    hr_estimates.append(round(server_hr.estimate(i + 1)))
    
# ------------------------------ Apple CMS Example (using aggregate_all and estimate_all) -------------------------

print("\n--- Simulating Aggregation for Apple CMS ---")
# This section aggregates all at once, so we spend the budget for all data points after privatization.
priv_data = [client_cms.privatise(item) for item in data]
server_cms.aggregate_all(priv_data)

# Record the budget for all the privatized items that were aggregated.
# This loop accounts for each client's contribution.
for _ in range(len(data)):
    accountant.spend(epsilon=epsilon, delta=0.0)

cms_estimates = server_cms.estimate_all(range(1, d + 1))
print(f"Finished aggregating {len(data)} items for Apple CMS. Total budget spent: {accountant.total()}")

# --------------------------------- PETINA CMS  -------------------------------------------------------------------
print("\n--- Simulating Aggregation for PETINA CMS ---")
# Hash functions must be consistent between client and server
petina_cms_hash_funcs = generate_hash_funcs(k, m)
use_hadamard_petina = False # Set to True to test the Hadamard variant

client_petina_cms = Client_PETINA_CMS(k, m, petina_cms_hash_funcs)
server_petina_cms = Server_PETINA_CMS(epsilon, k, m, petina_cms_hash_funcs, is_hadamard=use_hadamard_petina)

# --- Simulate data flow for PETINA CMS ---
for i, item in enumerate(data):
    prepared_data = client_petina_cms.prepare_item(item)
    server_petina_cms.aggregate_item(prepared_data)
    # Record the privacy budget spent for this client contribution
    accountant.spend(epsilon=epsilon, delta=0.0)
    # Print a progress update
    if (i + 1) % 5000 == 0:
        print(f"  Processed {i+1} clients. Total budget spent: {accountant.total()}")

# Finalize the sketch after all data has been aggregated
server_petina_cms.finalize_sketch()

# --- Simulate server-side estimation for PETINA CMS ---
petina_cms_estimates = []
for i in range(0, d):
    petina_cms_estimates.append(round(server_petina_cms.estimate(i + 1)))

# ------------------------------ PETINA Centralized CMS ----------------------------------------------------------
# Note: Centralized DP doesn't involve client-side privatization, so no LDP budget is spent here.
print("\n--- Running Centralized CMS (No LDP Budget Spend) ---")
use_hadamard_centralized = False # Set to True to test Hadamard variant
centralized_cms_estimate_func = centralized_count_mean_sketch(data, epsilon, k, m, is_hadamard=use_hadamard_centralized)
centralized_cms_estimates = []
for i in range(0, d):
    centralized_cms_estimates.append(round(centralized_cms_estimate_func(i + 1)))

# ------------------------------ Experiment Output (calculating variance) -------------------------
mse_arr = np.zeros(7) # Increased size for centralized CMS

for i in range(0, d):
    mse_arr[0] += (olh_estimates[i] - original_freq[i]) ** 2
    mse_arr[1] += (oue_estimates[i] - original_freq[i]) ** 2
    mse_arr[2] += (the_estimates[i] - original_freq[i]) ** 2
    mse_arr[3] += (hr_estimates[i] - original_freq[i]) ** 2
    mse_arr[4] += (cms_estimates[i] - original_freq[i]) ** 2
    mse_arr[5] += (petina_cms_estimates[i] - original_freq[i]) ** 2
    mse_arr[6] += (centralized_cms_estimates[i] - original_freq[i]) ** 2 # For centralized CMS

mse_arr = mse_arr / d

print("\n")
print("-" * 50)
print(
    "Experiment run on a dataset of size",
    len(data),
    "with d=",
    d,
    "and epsilon=",
    epsilon,
    "\n",
)
print("--- Frequency Estimation Performance (MSE) ---")
print("Optimised Local Hashing (OLH) Variance: ", mse_arr[0])
print("Optimised Unary Encoding (OUE) Variance: ", mse_arr[1])
print("Threshold Histogram Encoding (THE) Variance: ", mse_arr[2])
print("Hadamard response (HR) Variance:", mse_arr[3])
print("Apple CMS (LDP) Variance:", mse_arr[4])
print("PETINA CMS Variance:", mse_arr[5])
print("Centralized CMS Variance:", mse_arr[6])
print("\n")
print("--- Frequency Estimates ---")
print("Original Frequencies:", original_freq)
print("OLH Estimates:", olh_estimates)
print("OUE Estimates:", oue_estimates)
print("THE Estimates:", the_estimates)
print("HR Estimates:", hr_estimates)
print("CMS Estimates (LDP):", cms_estimates)
print("PETINA CMS Estimates:", petina_cms_estimates)
print("Centralized CMS Estimates:", centralized_cms_estimates)
print("Note: We round estimates to the nearest integer for display.")

# ------------------------------ Heavy Hitters - PEM Simulation -------------------------

print("\n--- Running Prefix Extending Method (PEM) to find heavy hitters ---")
pem_client = PEMClient(
    epsilon=3, start_length=2, max_string_length=6, fragment_length=2
)
pem_server = PEMServer(
    epsilon=3, start_length=2, max_string_length=6, fragment_length=2
)

s1 = "101101"
s2 = "111111"
s3 = "100000"
s4 = "101100"

print("Finding top 3 strings, where the alphabet is:", s1, s2, s3, s4)

data_hh = np.concatenate(([s1] * 8000, [s2] * 4000, [s3] * 1000, [s4] * 500))

for index, item in enumerate(data_hh):
    priv = pem_client.privatise(item)
    pem_server.aggregate(priv)
    # Record the privacy budget for each item in the heavy hitters simulation.
    accountant.spend(epsilon=pem_client.epsilon, delta=0.0)

# Can either specify top-k based or threshold based
# Threshold of 0.05 means we find any possible heavy hitters that have a frequency >= 5%
# Top-k of three means we try to find the top-3 most frequent strings
heavy_hitters, frequencies = pem_server.find_heavy_hitters(threshold=0.05)

print("\n--- Heavy Hitters Results ---")
print("Top strings found are:", heavy_hitters, " with frequencies", frequencies)

# --- Final Budget Summary ---
print("\n")
print("-" * 50)
print("Final Privacy Budget Summary")
print("-" * 50)
print(f"Total number of client contributions processed: {len(data) * 3 + len(data_hh)} (from all LDP and PEM simulations)")
print(f"Total privacy budget spent: {accountant.total()}")
print(f"Remaining budget (for 1 more query): {accountant.remaining(k=1)}")
print("-" * 50)