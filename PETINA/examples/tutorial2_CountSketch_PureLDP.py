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
#---- Import necessary libraries------------------------------------------
from pure_ldp.frequency_oracles import *
from pure_ldp.heavy_hitters import *
import numpy as np
from collections import Counter
import random
import math
from scipy.linalg import hadamard  # Make sure this is imported if you use Hadamard

# ---- Import from PETINA ----
from PETINA import generate_hash_funcs, Client_PETINA_CMS, Server_PETINA_CMS, centralized_count_mean_sketch
from PETINA import CSVec
# ---- CSVec import helper ----
import copy
import torch

# LARGEPRIME = 2**61 - 1

cache = {}

# ---- Client/Server wrappers for CSVec ----
class CSVecClient:
    def __init__(self, d, c, r):
        self.d = d
        self.c = c
        self.r = r

    def privatise(self, item):
        # Create a one-hot vector for the item
        one_hot_vec = torch.zeros(self.d)
        if 1 <= item <= self.d:
            one_hot_vec[item - 1] = 1
        
        # Sketch the one-hot vector
        csvec = CSVec(d=self.d, c=self.c, r=self.r)
        csvec.accumulateVec(one_hot_vec)
        return csvec.table

class CSVecServer:
    def __init__(self, d, c, r):
        self.d = d
        self.c = c
        self.r = r
        self.sketch = CSVec(d=self.d, c=self.c, r=self.r)

    def aggregate(self, sketch_table):
        self.sketch.accumulateTable(sketch_table)

    def aggregate_all(self, sketch_tables):
        for sketch_table in sketch_tables:
            self.aggregate(sketch_table)
            
    def estimate_all(self, items):
        # unSketch to recover the entire vector
        estimated_vector = self.sketch.unSketch(k=self.d)
        
        # Map indices back to items (1-based) and get estimates
        estimates = []
        for item in items:
            estimates.append(estimated_vector[item - 1].item())
            
        return estimates

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
k = 128  # 128 hash functions
m = 1024 # Each hash function maps to the domain {0, ... 1023}

server_cms = CMSServer(epsilon, k, m)
client_cms = CMSClient(epsilon, server_cms.get_hash_funcs(), m)

# ---- CSVec Parameters and Initialization ----
# c controls the magnitude of the error. If you want your error to be small on average, you need a large c.
# r controls the probability of a large error. If you want to be very confident that your error is small, you need a large r.
r_csvec = 10  # Number of rows in the sketch
c_csvec = 10  # Number of columns in the sketch (buckets)
client_csvec = CSVecClient(d=d, c=c_csvec, r=r_csvec)
server_csvec = CSVecServer(d=d, c=c_csvec, r=r_csvec)

# Simulate client-side privatisation + server-side aggregation for LDP methods
for item in data:
    priv_olh_data = client_olh.privatise(item)
    priv_oue_data = client_oue.privatise(item)
    priv_the_data = client_the.privatise(item)
    priv_hr_data = client_hr.privatise(item)
    priv_csvec_data = client_csvec.privatise(item) # Privatize data with CSVec

    server_olh.aggregate(priv_olh_data)
    server_oue.aggregate(priv_oue_data)
    server_the.aggregate(priv_the_data)
    server_hr.aggregate(priv_hr_data)
    server_csvec.aggregate(priv_csvec_data) # Aggregate CSVec sketch

# Note instead, we could use server.aggregate_all(list_of_privatised_data) see the Apple CMS example below

# Simulate server-side estimation
oue_estimates = []
olh_estimates = []
the_estimates = []
hr_estimates = []
csvec_estimates = []
mse_arr = np.zeros(8)  # Increased size for CSVec

for i in range(0, d):
    olh_estimates.append(round(server_olh.estimate(i + 1)))
    oue_estimates.append(round(server_oue.estimate(i + 1)))
    the_estimates.append(round(server_the.estimate(i + 1)))
    hr_estimates.append(round(server_hr.estimate(i + 1)))
    
# ------------------------------ Apple CMS Example (using aggregate_all and estimate_all) -------------------------

priv_data = [client_cms.privatise(item) for item in data]
server_cms.aggregate_all(priv_data)
cms_estimates = server_cms.estimate_all(range(1, d + 1))

# --------------------------------- PETINA CMS  -------------------------------------------------------------------
# Hash functions must be consistent between client and server
petina_cms_hash_funcs = generate_hash_funcs(k, m)
use_hadamard_petina = False  # Set to True to test the Hadamard variant

client_petina_cms = Client_PETINA_CMS(k, m, petina_cms_hash_funcs)
server_petina_cms = Server_PETINA_CMS(epsilon, k, m, petina_cms_hash_funcs, is_hadamard=use_hadamard_petina)

# --- Simulate data flow for PETINA CMS ---
for item in data:
    prepared_data = client_petina_cms.prepare_item(item)
    server_petina_cms.aggregate_item(prepared_data)

# Finalize the sketch after all data has been aggregated
server_petina_cms.finalize_sketch()

# --- Simulate server-side estimation for PETINA CMS ---
petina_cms_estimates = []
for i in range(0, d):
    petina_cms_estimates.append(round(server_petina_cms.estimate(i + 1)))


# ------------------------------ PETINA Centralized CMS ----------------------------------------------------------


use_hadamard_centralized = False  # Set to True to test Hadamard variant
centralized_cms_estimate_func = centralized_count_mean_sketch(data, epsilon, k, m, is_hadamard=use_hadamard_centralized)
centralized_cms_estimates = []
for i in range(0, d):
    centralized_cms_estimates.append(round(centralized_cms_estimate_func(i + 1)))

# ------------------------------ CSVec Estimation --------------------------------------------------------------
# We estimate all items at once.
csvec_estimates = server_csvec.estimate_all(range(1, d + 1))
csvec_estimates = [round(est) for est in csvec_estimates]

# ------------------------------ Experiment Output (calculating variance) -------------------------

for i in range(0, d):
    mse_arr[0] += (olh_estimates[i] - original_freq[i]) ** 2
    mse_arr[1] += (oue_estimates[i] - original_freq[i]) ** 2
    mse_arr[2] += (the_estimates[i] - original_freq[i]) ** 2
    mse_arr[3] += (hr_estimates[i] - original_freq[i]) ** 2
    mse_arr[4] += (cms_estimates[i] - original_freq[i]) ** 2
    mse_arr[5] += (petina_cms_estimates[i] - original_freq[i]) ** 2
    mse_arr[6] += (centralized_cms_estimates[i] - original_freq[i]) ** 2  # For centralized CMS
    mse_arr[7] += (csvec_estimates[i] - original_freq[i]) ** 2  # For CSVec

mse_arr = mse_arr / d

print("\n")
print(
    "Experiment run on a dataset of size",
    len(data),
    "with d=",
    d,
    "and epsilon=",
    epsilon,
    "\n",
)
print("Optimised Local Hashing (OLH) Variance: ", mse_arr[0])
print("Optimised Unary Encoding (OUE) Variance: ", mse_arr[1])
print("Threshold Histogram Encoding (THE) Variance: ", mse_arr[2])
print("Hadamard response (HR) Variance:", mse_arr[3])
print(sum(hr_estimates))
print("Apple CMS (LDP) Variance:", mse_arr[4])
print("PETINA CMS Variance:", mse_arr[5])  # New line for PETINA CMS
print("Centralized CMS Variance:", mse_arr[6])  # New line for centralized CMS
print("CSVec Variance:", mse_arr[7])  # New line for CSVec
print("\n")
print("Original Frequencies:", original_freq)
print("OLH Estimates:", olh_estimates)
print("OUE Estimates:", oue_estimates)
print("THE Estimates:", the_estimates)
print("HR Estimates:", hr_estimates)
print("CMS Estimates (LDP):", cms_estimates)
print("PETINA CMS Estimates:", petina_cms_estimates)  # New line for PETINA CMS
print("Centralized CMS Estimates:", centralized_cms_estimates)  # New line for centralized CMS
print("CSVec Estimates:", csvec_estimates)  # New line for CSVec
print("Note: We round estimates to the nearest integer")
print("\n")

# ------------------------------ Heavy Hitters - PEM Simulation -------------------------

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

print("\nRunning Prefix Extending Method (PEM) to find heavy hitters")
print("Finding top 3 strings, where the alphabet is:", s1, s2, s3, s4)

data = np.concatenate(([s1] * 8000, [s2] * 4000, [s3] * 1000, [s4] * 500))

for index, item in enumerate(data):
    priv = pem_client.privatise(item)
    pem_server.aggregate(priv)

# Can either specify top-k based or threshold based
# Threshold of 0.05 means we find any possible heavy hitters that have a frequency >= 5%
# Top-k of three means we try to find the top-3 most frequent strings

heavy_hitters, frequencies = pem_server.find_heavy_hitters(threshold=0.05)
print("Top strings found are:", heavy_hitters, " with frequencies", frequencies)