import math
import random
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import warnings
from numbers import Real, Integral
import warnings # Required for the IBM BudgetAccountant's internal warnings
from numbers import Real, Integral # Required for check_epsilon_delta and BudgetAccountant
from PETINA.Data_Conversion_Helper import TypeConverter
from PETINA.package.csvec.csvec import CSVec
from PETINA import BudgetAccountant, BudgetError

# --- Mock diffprivlib components for BudgetAccountant ---
# These are typically imported from diffprivlib.utils and diffprivlib.validation
# but are included here for a self-contained example with the provided BudgetAccountant.

# class Budget(tuple):
#     """A simple class to represent an (epsilon, delta) privacy budget."""
#     def __init__(self, epsilon: float, delta: float):
#         self.epsilon = float(epsilon)
#         self.delta = float(delta)

#     def __repr__(self):
#         return f"(epsilon={self.epsilon}, delta={self.delta})"

#     def __ge__(self, other: 'Budget') -> bool:
#         """Checks if this budget is greater than or equal to another budget."""
#         return self.epsilon >= other.epsilon and self.delta >= other.delta

#     def __eq__(self, other: 'Budget') -> bool:
#         """Checks if this budget is equal to another budget."""
#         return self.epsilon == other.epsilon and self.delta == other.delta

#     def __add__(self, other: 'Budget') -> 'Budget':
#         """Adds two budgets (simple sum, not composition)."""
#         return Budget(self.epsilon + other.epsilon, self.delta + other.delta)

# class BudgetError(Exception):
#     """Custom exception for privacy budget exhaustion."""
#     pass

# def check_epsilon_delta(epsilon: float, delta: float):
#     """
#     A simple validation function for epsilon and delta values.
#     Raises ValueError if values are invalid.
#     """
#     if not isinstance(epsilon, (float, int)) or epsilon < 0:
#         raise ValueError(f"Epsilon must be a non-negative float. Got {epsilon}")
#     if not isinstance(delta, (float, int)) or not (0 <= delta <= 1):
#         raise ValueError(f"Delta must be a float between 0 and 1 inclusive. Got {delta}")

# # --- IBM BudgetAccountant Class (as provided by user) ---
# # MIT License
# #
# # Copyright (C) IBM Corporation 2020
# #
# # Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# # documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# # rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# # persons to whom the Software is furnished to do so, subject to the following conditions:
# #
# # The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# # Software.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# # WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# # TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.
# """
# Privacy budget accountant for differential privacy
# """
# from numbers import Integral

# class BudgetAccountant:
#     """Privacy budget accountant for differential privacy.
#     This class creates a privacy budget accountant to track privacy spend across queries and other data accesses.  Once
#     initialised, the BudgetAccountant stores each privacy spend and iteratively updates the total budget spend, raising
#     an error when the budget ceiling (if specified) is exceeded.  The accountant can be initialised without any maximum
#     budget, to enable users track the total privacy spend of their actions without hindrance.
#     Implements the accountant rules as given in [KOV17]_.
#     Parameters
#     ----------
#     epsilon : float, default: infinity
#         Epsilon budget ceiling of the accountant.
#     delta : float, default: 1.0
#         Delta budget ceiling of the accountant.
#     slack : float, default: 0.0
#         Slack allowed in delta spend.  Greater slack may reduce the overall epsilon spend.
#     spent_budget : list of tuples of the form (epsilon, delta), optional
#         List of tuples of pre-existing budget spends.  Allows for a new accountant to be initialised with spends
#         extracted from a previous instance.
#     """
#     _default = None

#     def __init__(self, epsilon=float("inf"), delta=1.0, slack=0.0, spent_budget=None):
#         check_epsilon_delta(epsilon, delta)
#         self.__epsilon = epsilon
#         self.__min_epsilon = 0 if epsilon == float("inf") else epsilon * 1e-14
#         self.__delta = delta
#         self.__spent_budget = []
#         self.slack = slack

#         if spent_budget is not None:
#             if not isinstance(spent_budget, list):
#                 raise TypeError("spent_budget must be a list")

#             for _epsilon, _delta in spent_budget:
#                 self.spend(_epsilon, _delta)

#     def __repr__(self, n_budget_max=5):
#         params = []
#         if self.epsilon != float("inf"):
#             params.append(f"epsilon={self.epsilon}")

#         if self.delta != 1:
#             params.append(f"delta={self.delta}")

#         if self.slack > 0:
#             params.append(f"slack={self.slack}")

#         if self.spent_budget:
#             if len(self.spent_budget) > n_budget_max:
#                 params.append("spent_budget=" + str(self.spent_budget[:n_budget_max] + ["..."]).replace("'", ""))
#             else:
#                 params.append("spent_budget=" + str(self.spent_budget))

#         return "BudgetAccountant(" + ", ".join(params) + ")"

#     def __enter__(self):
#         self.old_default = self.pop_default()
#         self.set_default()
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.pop_default()

#         if self.old_default is not None:
#             self.old_default.set_default()
#         del self.old_default

#     def __len__(self):
#         return len(self.spent_budget)

#     @property
#     def slack(self):
#         """Slack parameter for composition.
#         """
#         return self.__slack

#     @slack.setter
#     def slack(self, slack):
#         if not 0 <= slack <= self.delta:
#             raise ValueError(f"Slack must be between 0 and delta ({self.delta}), inclusive. Got {slack}.")

#         epsilon_spent, delta_spent = self.total(slack=slack)

#         if self.epsilon < epsilon_spent or self.delta < delta_spent:
#             raise BudgetError(f"Privacy budget will be exceeded by changing slack to {slack}.")

#         self.__slack = slack

#     @property
#     def spent_budget(self):
#         """List of tuples of the form (epsilon, delta) of spent privacy budget.
#         """
#         return self.__spent_budget.copy()

#     @property
#     def epsilon(self):
#         """Epsilon privacy ceiling of the accountant.
#         """
#         return self.__epsilon

#     @property
#     def delta(self):
#         """Delta privacy ceiling of the accountant.
#         """
#         return self.__delta

#     def total(self, spent_budget=None, slack=None):
#         """Returns the total current privacy spend.
#         `spent_budget` and `slack` can be specified as parameters, otherwise the class values will be used.
#         Parameters
#         ----------
#         spent_budget : list of tuples of the form (epsilon, delta), optional
#             List of tuples of budget spends.  If not provided, the accountant's spends will be used.
#         slack : float, optional
#             Slack in delta for composition.  If not provided, the accountant's slack will be used.
#         Returns
#         -------
#         epsilon : float
#             Total epsilon spend.
#         delta : float
#             Total delta spend.
#         """
#         if spent_budget is None:
#             spent_budget = self.spent_budget
#         else:
#             for epsilon, delta in spent_budget:
#                 check_epsilon_delta(epsilon, delta)

#         if slack is None:
#             slack = self.slack
#         elif not 0 <= slack <= self.delta:
#             raise ValueError(f"Slack must be between 0 and delta ({self.delta}), inclusive. Got {slack}.")

#         epsilon_sum, epsilon_exp_sum, epsilon_sq_sum = 0, 0, 0

#         for epsilon, _ in spent_budget:
#             epsilon_sum += epsilon
#             epsilon_exp_sum += (1 - np.exp(-epsilon)) * epsilon / (1 + np.exp(-epsilon))
#             epsilon_sq_sum += epsilon ** 2

#         total_epsilon_naive = epsilon_sum
#         total_delta = self.__total_delta_safe(spent_budget, slack)

#         if slack == 0:
#             return Budget(total_epsilon_naive, total_delta)

#         # Advanced composition from Kairouz et al. (2017)
#         total_epsilon_kov = epsilon_exp_sum + np.sqrt(2 * epsilon_sq_sum * np.log(np.exp(1) + np.sqrt(epsilon_sq_sum) / slack))
        
#         return Budget(min(total_epsilon_naive, total_epsilon_kov), total_delta)

#     def check(self, epsilon, delta):
#         """Checks if the provided (epsilon,delta) can be spent without exceeding the accountant's budget ceiling.
#         Parameters
#         ----------
#         epsilon : float
#             Epsilon budget spend to check.
#         delta : float
#             Delta budget spend to check.
#         Returns
#         -------
#         bool
#             True if the budget can be spent, otherwise a :class:`.BudgetError` is raised.
#         Raises
#         ------
#         BudgetError
#             If the specified budget spend will result in the budget ceiling being exceeded.
#         """
#         check_epsilon_delta(epsilon, delta)
#         if self.epsilon == float("inf") and self.delta == 1:
#             return True

#         if 0 < epsilon < self.__min_epsilon:
#             raise ValueError(f"Epsilon must be at least {self.__min_epsilon} if non-zero, got {epsilon}.")

#         spent_budget = self.spent_budget + [(epsilon, delta)]

#         if Budget(self.epsilon, self.delta) >= self.total(spent_budget=spent_budget):
#             return True

#         raise BudgetError(f"Privacy spend of ({epsilon},{delta}) not permissible; will exceed remaining privacy budget."
#                           f" Use {self.__class__.__name__}.{self.remaining.__name__}() to check remaining budget.")

#     def remaining(self, k=1):
#         """Calculates the budget that remains to be spent.
#         Calculates the privacy budget that can be spent on `k` queries.  Spending this budget on `k` queries will
#         match the budget ceiling, assuming no floating point errors.
#         Parameters
#         ----------
#         k : int, default: 1
#             The number of queries for which to calculate the remaining budget.
#         Returns
#         -------
#         epsilon : float
#             Total epsilon spend remaining for `k` queries.
#         delta : float
#             Total delta spend remaining for `k` queries.
#         """
#         if not isinstance(k, Integral):
#             raise TypeError(f"k must be integer-valued, got {type(k)}.")
#         if k < 1:
#             raise ValueError(f"k must be at least 1, got {k}.")

#         _, spent_delta = self.total()
#         delta_k = 1 - ((1 - self.delta) / (1 - spent_delta)) ** (1 / k) if spent_delta < 1.0 else 1.0
#         delta = min(1.0, delta_k)

#         lower = 0
#         upper = self.epsilon
#         tolerance = self.epsilon * 1e-6
#         max_iterations = 100
        
#         for _ in range(max_iterations):
#             mid = (upper + lower) / 2
#             spent_budget_for_check = self.spent_budget + [(mid, 0)] * k
#             current_total_epsilon, _ = self.total(spent_budget=spent_budget_for_check, slack=self.slack)

#             if abs(upper - lower) < tolerance:
#                 break

#             if current_total_epsilon >= self.epsilon:
#                 upper = mid
#             else:
#                 lower = mid

#         epsilon = (upper + lower) / 2

#         return Budget(epsilon, delta)

#     def spend(self, epsilon, delta):
#         """Spend the given privacy budget.
#         Instructs the accountant to spend the given epsilon and delta privacy budget, while ensuring the target budget
#         is not exceeded.
#         Parameters
#         ----------
#         epsilon : float
#             Epsilon privacy budget to spend.
#         delta : float
#             Delta privacy budget to spend.
#         Returns
#         -------
#         self : BudgetAccountant
#         """
#         self.check(epsilon, delta)
#         self.__spent_budget.append(Budget(epsilon, delta))
#         return self

#     @staticmethod
#     def __total_delta_safe(spent_budget, slack):
#         """
#         Calculate total delta spend of `spent_budget`, with special consideration for floating point arithmetic.
#         Should yield greater precision, especially for a large number of budget spends with very small delta.
#         Parameters
#         ----------
#         spent_budget: list of tuples of the form (epsilon, delta)
#             List of budget spends, for which the total delta spend is to be calculated.
#         slack: float
#             Delta slack parameter for composition of spends.
#         Returns
#         -------
#         float
#             Total delta spend.
#         """
#         delta_spend = [slack]
#         for _, delta in spent_budget:
#             delta_spend.append(delta)
#         delta_spend.sort()

#         prod = 0
#         for delta in delta_spend:
#             prod += delta - prod * delta

#         return prod

#     @staticmethod
#     def load_default(accountant):
#         """Loads the default privacy budget accountant if none is supplied, otherwise checks that the supplied
#         accountant is a BudgetAccountant class.
#         An accountant can be set as the default() method.  If no default has been set, a default
#         is created.
#         Parameters
#         ----------
#         accountant : BudgetAccountant or None
#             The supplied budget accountant.  If None, the default accountant is returned.
#         Returns
#         -------
#         default : BudgetAccountant
#             Returns a working BudgetAccountant, either the supplied `accountant` or the existing default.
#         """
#         if accountant is None:
#             if BudgetAccountant._default is None:
#                 BudgetAccountant._default = BudgetAccountant()

#             return BudgetAccountant._default

#         if not isinstance(accountant, BudgetAccountant):
#             raise TypeError(f"Accountant must be of type BudgetAccountant, got {type(accountant)}")

#         return accountant

#     def set_default(self):
#         """Sets the current accountant to be the default when running functions and queries with diffprivlib.
#         Returns
#         -------
#         self : BudgetAccountant
#         """
#         BudgetAccountant._default = self
#         return self

#     @staticmethod
#     def pop_default():
#         """Pops the default BudgetAccountant from the class and returns it to the user.
#         Returns
#         -------
#         default : BudgetAccountant
#             Returns the existing default BudgetAccountant.
#         """
#         default = BudgetAccountant._default
#         BudgetAccountant._default = None
#         return default

# # --- TypeConverter for handling various data types ---
# class TypeConverter:
#     """
#     A utility class to convert input data (list, numpy array, or torch.Tensor)
#     to a flattened torch.Tensor and restore it to its original type and shape.
#     """
#     def __init__(self, data):
#         self.original_type = type(data)
#         self.original_shape = None
#         self.is_torch_tensor = False

#         if isinstance(data, torch.Tensor):
#             self.is_torch_tensor = True
#             self.original_shape = data.shape
#             self.flattened_data = data.flatten()
#         elif isinstance(data, np.ndarray):
#             self.original_shape = data.shape
#             self.flattened_data = torch.from_numpy(data.flatten()).float() # Convert to torch.Tensor
#         elif isinstance(data, list):
#             # For lists, assume it's already flat or handle simple nested lists.
#             # Convert to numpy then to torch.Tensor for consistency.
#             flat_list = [item for sublist in data for item in (sublist if isinstance(sublist, list) else [sublist])] if any(isinstance(i, list) for i in data) else data
#             self.flattened_data = torch.tensor(flat_list, dtype=torch.float32)
#             self.original_shape = (len(self.flattened_data),)
#         else:
#             raise TypeError("Unsupported data type for TypeConverter. Must be list, numpy.ndarray, or torch.Tensor.")

#     def get(self) -> tuple[torch.Tensor, tuple]:
#         """Returns the flattened data (as torch.Tensor) and its original shape."""
#         return self.flattened_data, self.original_shape

#     def restore(self, flattened_data_list: list) -> list | np.ndarray | torch.Tensor:
#         """Restores flattened data (from a list) to its original type and shape."""
#         restored_tensor = torch.tensor(flattened_data_list, dtype=torch.float32).reshape(self.original_shape)
#         if self.is_torch_tensor:
#             return restored_tensor
#         elif self.original_type == np.ndarray:
#             return restored_tensor.numpy()
#         else: # list
#             return flattened_data_list # Note: won't restore complex nested list structure

# # --- CSVec Class Definition (Count Sketch Implementation) ---
# # This is the robust Count Sketch implementation used for gradient compression.
# # In a real PETINA library, this would be imported from a dedicated package.
# LARGEPRIME = 2**61-1
# _csvec_cache = {} # Global cache for precomputed hash functions

# class CSVec(object):
#     """
#     Count Sketch of a vector. This class efficiently computes the count sketch
#     of an input vector and supports operations like accumulation and reconstruction.
#     """
#     def __init__(self, d: int, c: int, r: int, doInitialize: bool = True, device: torch.device | str | None = None, numBlocks: int = 1):
#         """
#         Constructor for CSVec.
#         Args:
#             d: Original dimensionality of the vector to be sketched.
#             c: Number of columns (buckets) in the sketch table.
#             r: Number of rows in the sketch table (number of independent hash functions).
#             doInitialize: If False, skips hash function setup (used by deepcopy).
#             device: Which device to use (cuda or cpu).
#             numBlocks: Memory optimization for hash functions.
#         """
#         global _csvec_cache
#         self.r = r
#         self.c = c
#         self.d = int(d)
#         self.numBlocks = numBlocks

#         if device is None:
#             device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.device = torch.device(device)

#         if not doInitialize: return
#         self.table = torch.zeros((r, c), device=self.device)

#         cacheKey = (d, c, r, numBlocks, str(self.device))
#         if cacheKey in _csvec_cache:
#             cached_data = _csvec_cache[cacheKey]
#             self.signs = cached_data["signs"]
#             self.buckets = cached_data["buckets"]
#             if self.numBlocks > 1:
#                 self.blockSigns = cached_data["blockSigns"]
#                 self.blockOffsets = cached_data["blockOffsets"]
#             return

#         rand_state = torch.random.get_rng_state()
#         torch.random.manual_seed(42) # Fixed seed for hash generation
#         hashes = torch.randint(0, LARGEPRIME, (r, 6), dtype=torch.int64, device="cpu")
        
#         if self.numBlocks > 1:
#             nTokens = self.d // numBlocks
#             if self.d % numBlocks != 0: nTokens += 1
#             self.blockSigns = (torch.randint(0, 2, size=(self.numBlocks,), device=self.device) * 2 - 1).float()
#             self.blockOffsets = torch.randint(0, self.c, size=(self.numBlocks,), device=self.device).long()
#         else:
#             assert(numBlocks == 1)
#             nTokens = self.d
        
#         torch.random.set_rng_state(rand_state)

#         tokens = torch.arange(nTokens, dtype=torch.int64, device="cpu").reshape((1, nTokens))

#         h1, h2, h3, h4 = hashes[:,2:3], hashes[:,3:4], hashes[:,4:5], hashes[:,5:6]
#         self.signs = (((h1 * tokens + h2) * tokens + h3) * tokens + h4)
#         self.signs = ((self.signs % LARGEPRIME % 2) * 2 - 1).float().to(self.device)

#         h1, h2 = hashes[:,0:1], hashes[:,1:2]
#         self.buckets = ((h1 * tokens) + h2) % LARGEPRIME % self.c
#         self.buckets = self.buckets.to(self.device).long()

#         _csvec_cache[cacheKey] = {"signs": self.signs, "buckets": self.buckets}
#         if numBlocks > 1:
#             _csvec_cache[cacheKey].update({"blockSigns": self.blockSigns, "blockOffsets": self.blockOffsets})

#     def zero(self): self.table.zero_()
#     def cpu_(self): self.device = torch.device("cpu"); self.table = self.table.cpu()
#     def cuda_(self, device: torch.device | str = "cuda"): self.device = torch.device(device); self.table = self.table.cuda(device)
#     def half_(self): self.table = self.table.half()
#     def float_(self): self.table = self.table.float()

#     def __deepcopy__(self, memodict={}):
#         newCSVec = CSVec(d=self.d, c=self.c, r=self.r, doInitialize=False, device=self.device, numBlocks=self.numBlocks)
#         newCSVec.table = copy.deepcopy(self.table)
#         global _csvec_cache
#         cachedVals = _csvec_cache[(self.d, self.c, self.r, self.numBlocks, str(self.device))]
#         newCSVec.signs = cachedVals["signs"]
#         newCSVec.buckets = cachedVals["buckets"]
#         if self.numBlocks > 1:
#             newCSVec.blockSigns = cachedVals["blockSigns"]
#             newCSVec.blockOffsets = cachedVals["blockOffsets"]
#         return newCSVec

#     def __imul__(self, other: int | float):
#         if isinstance(other, (int, float)): self.table.mul_(other)
#         else: raise ValueError(f"Can't multiply a CSVec by {other}")
#         return self

#     def __truediv__(self, other: int | float):
#         if isinstance(other, (int, float)): self.table.div_(other)
#         else: raise ValueError(f"Can't divide a CSVec by {other}")
#         return self

#     def __add__(self, other: 'CSVec'):
#         returnCSVec = copy.deepcopy(self)
#         returnCSVec += other
#         return returnCSVec

#     def __iadd__(self, other: 'CSVec'):
#         if isinstance(other, CSVec):
#             assert(self.d == other.d and self.c == other.c and self.r == other.r and
#                    self.device == other.device and self.numBlocks == other.numBlocks)
#             self.table += other.table
#         else:
#             raise ValueError("Can't add this to a CSVec: {}".format(other))
#         return self

#     def accumulateTable(self, table: torch.Tensor):
#         if table.size() != self.table.size():
#             raise ValueError(f"Passed in table has size {table.size()}, expecting {self.table.size()}")
#         self.table += table

#     def accumulateVec(self, vec: torch.Tensor):
#         """Sketches a vector and adds the result to the internal sketch table."""
#         assert(len(vec.size()) == 1 and vec.size()[0] == self.d)
#         for r_idx in range(self.r):
#             buckets = self.buckets[r_idx,:].to(self.device)
#             signs = self.signs[r_idx,:].to(self.device)
#             for blockId in range(self.numBlocks):
#                 start = blockId * buckets.size()[0]
#                 end = min((blockId + 1) * buckets.size()[0], self.d)
#                 offsetBuckets = buckets[:end-start].clone()
#                 offsetSigns = signs[:end-start].clone()
#                 if self.numBlocks > 1:
#                     offsetBuckets += self.blockOffsets[blockId]
#                     offsetBuckets %= self.c
#                     offsetSigns *= self.blockSigns[blockId]
                
#                 vec_slice = vec[start:end]
#                 if vec_slice.device != self.device: vec_slice = vec_slice.to(self.device)

#                 self.table[r_idx,:] += torch.bincount(
#                                         input=offsetBuckets,
#                                         weights=offsetSigns * vec_slice,
#                                         minlength=self.c
#                                        )

#     def _findHHK(self, k: int):
#         vals = self._findAllValues()
#         outVals = torch.zeros(k, device=vals.device)
#         HHs = torch.zeros(k, device=vals.device, dtype=torch.long)
#         torch.topk(vals**2, k, sorted=False, out=(outVals, HHs))
#         return HHs, vals[HHs]

#     def _findHHThr(self, thr: float):
#         vals = self._findAllValues()
#         HHs = vals.abs() >= thr
#         return HHs, vals[HHs]

#     def _findValues(self, coords: torch.Tensor):
#         assert(self.numBlocks == 1)
#         d_coords = coords.size()[0]
#         vals = torch.zeros(self.r, d_coords, device=self.device)
#         for r_idx in range(self.r):
#             vals[r_idx] = (self.table[r_idx, self.buckets[r_idx, coords]] * self.signs[r_idx, coords])
#         return vals.median(dim=0)[0]

#     def _findAllValues(self) -> torch.Tensor:
#         """Reconstructs the entire original vector from the sketch."""
#         if self.numBlocks == 1:
#             vals = torch.zeros(self.r, self.d, device=self.device)
#             for r_idx in range(self.r):
#                 vals[r_idx] = (self.table[r_idx, self.buckets[r_idx,:]] * self.signs[r_idx,:])
#             return vals.median(dim=0)[0]
#         else:
#             medians = torch.zeros(self.d, device=self.device)
#             for blockId in range(self.numBlocks):
#                 start = blockId * self.buckets.size()[1]
#                 end = min((blockId + 1) * self.buckets.size()[1], self.d)
#                 vals = torch.zeros(self.r, end-start, device=self.device)
#                 for r_idx in range(self.r):
#                     buckets = self.buckets[r_idx, :end-start]
#                     signs = self.signs[r_idx, :end-start]
#                     offsetBuckets = buckets + self.blockOffsets[blockId]
#                     offsetBuckets %= self.c
#                     offsetSigns = signs * self.blockSigns[blockId]
#                     vals[r_idx] = (self.table[r_idx, offsetBuckets] * offsetSigns)
#                 medians[start:end] = vals.median(dim=0)[0]
#             return medians

#     def _findHHs(self, k: int | None = None, thr: float | None = None):
#         assert((k is None) != (thr is None))
#         if k is not None: return self._findHHK(k)
#         else: return self._findHHThr(thr)

#     def unSketch(self, k: int | None = None, epsilon: float | None = None) -> torch.Tensor:
#         """Performs heavy-hitter recovery or full vector reconstruction."""
#         if epsilon is None: thr = None
#         else: thr = epsilon * self.l2estimate()
        
#         if k is not None and k == self.d and epsilon is None: return self._findAllValues()
#         else:
#             hhs = self._findHHs(k=k, thr=thr)
#             unSketched = torch.zeros(self.d, device=self.device)
#             unSketched[hhs[0]] = hhs[1]
#             return unSketched

#     def l2estimate(self) -> float:
#         """Returns an estimate of the L2 norm of the sketched vector."""
#         return np.sqrt(torch.median(torch.sum(self.table**2,1)).item())

#     @classmethod
#     def median(cls, csvecs: list['CSVec']):
#         """Computes the median of multiple CSVec instances."""
#         d, c, r, device, numBlocks = csvecs[0].d, csvecs[0].c, csvecs[0].r, csvecs[0].device, csvecs[0].numBlocks
#         for csvec in csvecs:
#             assert(csvec.d == d and csvec.c == c and csvec.r == r and
#                    csvec.device == device and csvec.numBlocks == numBlocks)
#         tables = [csvec.table for csvec in csvecs]
#         med = torch.median(torch.stack(tables), dim=0)[0]
#         returnCSVec = copy.deepcopy(csvecs[0])
#         returnCSVec.table = med
#         return returnCSVec

# --- DP_Mechanisms Class ---
# This class contains functions for applying different types of DP noise,
# including the applyCountSketch method.
class DP_Mechanisms:
    @staticmethod
    def applyDPGaussian(domain: np.ndarray, delta: float = 1e-5, epsilon: float = 0.1, gamma: float = 1.0, accountant: BudgetAccountant | None = None) -> np.ndarray:
        """
        Applies Gaussian noise to the input NumPy array for differential privacy,
        and optionally tracks budget via a BudgetAccountant.
        This function expects and returns a NumPy array.
        """
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * gamma / epsilon
        privatized = domain + np.random.normal(loc=0, scale=sigma, size=domain.shape) * 1.572 # Retaining *1.572 from your original code
        
        if accountant is not None:
            accountant.spend(epsilon, delta)
        return privatized

    @staticmethod
    def applyDPLaplace(domain: np.ndarray, sensitivity: float = 1, epsilon: float = 0.01, gamma: float = 1, accountant: BudgetAccountant | None = None) -> np.ndarray:
        """
        Applies Laplace noise to the input NumPy array for differential privacy.
        Tracks privacy budget with an optional BudgetAccountant.
        This function expects and returns a NumPy array.
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be > 0 for Laplace mechanism.")
        scale = sensitivity * gamma / epsilon
        privatized = domain + np.random.laplace(loc=0, scale=scale, size=domain.shape)

        if accountant is not None:
            cost_epsilon, cost_delta = epsilon, 0.0 # Laplace mechanism typically has delta=0
            accountant.spend(cost_epsilon, cost_delta)
            
        return privatized

    @staticmethod
    def applyCountSketch(
        domain: list | np.ndarray | torch.Tensor,
        num_rows: int,
        num_cols: int,
        epsilon: float,
        delta: float,
        mechanism: str = "gaussian",
        sensitivity: float = 1.0,
        gamma: float = 0.01,
        num_blocks: int = 1,
        device: torch.device | str | None = None,
        accountant: BudgetAccountant | None = None
        ) -> list | np.ndarray | torch.Tensor:
        """
        Applies Count Sketch to the input data, then adds differential privacy
        noise to the sketched representation, and finally reconstructs the data.
        Consumes budget from the provided BudgetAccountant.
        """
        converter = TypeConverter(domain)
        flattened_data_tensor, original_shape = converter.get()

        # Ensure tensor
        if not isinstance(flattened_data_tensor, torch.Tensor):
            flattened_data_tensor = torch.tensor(flattened_data_tensor, dtype=torch.float32)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)

        flattened_data_tensor = flattened_data_tensor.to(device)

        csvec_instance = CSVec(
            d=flattened_data_tensor.numel(),
            c=num_cols,
            r=num_rows,
            numBlocks=num_blocks,
            device=device
        )

        csvec_instance.accumulateVec(flattened_data_tensor)

        sketched_table_np = csvec_instance.table.detach().cpu().numpy()

        if mechanism == "gaussian":
            noisy_sketched_table_np = DP_Mechanisms.applyDPGaussian(
                sketched_table_np, delta=delta, epsilon=epsilon, gamma=gamma, accountant=accountant
            )
        elif mechanism == "laplace":
            noisy_sketched_table_np = DP_Mechanisms.applyDPLaplace(
                sketched_table_np, sensitivity=sensitivity, epsilon=epsilon, gamma=gamma, accountant=accountant
            )
        else:
            raise ValueError(f"Unsupported DP mechanism for Count Sketch: {mechanism}. Choose 'gaussian' or 'laplace'.")

        csvec_instance.table = torch.tensor(noisy_sketched_table_np, dtype=torch.float32).to(device)
        reconstructed_noisy_data = csvec_instance._findAllValues()

        return converter.restore(reconstructed_noisy_data.tolist())

# File: PETINA/PETINA/examples/4_ML_CIFAR_10_No_MA.py
# ======================================================
#         CIFAR-10 Training with Differential Privacy
# ======================================================

# --- Set seeds for reproducibility ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed=42
set_seed(seed)

# --- Setup device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# --- Load CIFAR-10 dataset ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

batch_size = 1024
testloader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# --- Simple CNN Model ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Evaluation ---
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

# --- DP noise wrappers with budget accounting ---
# These functions now correctly handle the conversion to/from NumPy for DP_Mechanisms
def apply_laplace_with_budget(grad: torch.Tensor, sensitivity: float, epsilon: float, gamma: float, accountant: BudgetAccountant) -> torch.Tensor:
    grad_np = grad.cpu().numpy() # Convert PyTorch Tensor to NumPy array
    noisy_np = DP_Mechanisms.applyDPLaplace(grad_np, sensitivity=sensitivity, epsilon=epsilon, gamma=gamma, accountant=accountant)
    return torch.tensor(noisy_np, dtype=torch.float32).to(device) # Convert NumPy array back to PyTorch Tensor

def apply_gaussian_with_budget(grad: torch.Tensor, delta: float, epsilon: float, gamma: float, accountant: BudgetAccountant) -> torch.Tensor:
    grad_np = grad.cpu().numpy() # Convert PyTorch Tensor to NumPy array
    noisy_np = DP_Mechanisms.applyDPGaussian(grad_np, delta=delta, epsilon=epsilon, gamma=gamma, accountant=accountant)
    return torch.tensor(noisy_np, dtype=torch.float32).to(device) # Convert NumPy array back to PyTorch Tensor

# --- Federated Learning Components ---

class FederatedClient:
    def __init__(self, client_id: int, train_data: torch.utils.data.Dataset, device: torch.device,
                 dp_type: str | None, dp_params: dict, use_count_sketch: bool, sketch_params: dict | None,
                 accountant: BudgetAccountant, epochs_per_round: int, batch_size: int):
        self.client_id = client_id
        self.trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.device = device
        self.dp_type = dp_type
        self.dp_params = dp_params
        self.use_count_sketch = use_count_sketch
        self.sketch_params = sketch_params
        self.accountant = accountant
        self.epochs_per_round = epochs_per_round
        self.local_model = SimpleCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=0.01, momentum=0.9)
        self.scaler = torch.amp.GradScaler('cuda' if self.device.type == 'cuda' else 'cpu')
        self.mechanism_map = {
            'gaussian': "gaussian",
            'laplace': "laplace"
        }

    def set_global_model(self, global_model_state_dict: dict):
        """Sets the client's local model to the state of the global model."""
        self.local_model.load_state_dict(global_model_state_dict)

    def get_model_parameters(self) -> dict:
        """Returns the current state dictionary of the local model."""
        return self.local_model.state_dict()

    def train_local(self) -> dict:
        """
        Performs local training on the client's data and returns the privatized
        model updates (or parameters).
        """
        self.local_model.train()
        for e in range(self.epochs_per_round):
            for inputs, targets in self.trainloader:
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.device.type):
                    outputs = self.local_model(inputs)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()

                if self.dp_type is not None:
                    self.scaler.unscale_(self.optimizer) 
                    
                    if self.use_count_sketch:
                        grad_list = [p.grad.view(-1) for p in self.local_model.parameters() if p.grad is not None]
                        if not grad_list: continue
                        flat_grad = torch.cat(grad_list)

                        mechanism_str = self.mechanism_map.get(self.dp_type)
                        if mechanism_str is None:
                            raise ValueError(f"Unsupported DP noise type '{self.dp_type}' for Count Sketch DP.")
                        
                        privatized_grad_tensor = DP_Mechanisms.applyCountSketch(
                            domain=flat_grad,
                            num_rows=self.sketch_params['d'],
                            num_cols=self.sketch_params['w'],
                            epsilon=self.dp_params['epsilon'],
                            delta=self.dp_params['delta'],
                            mechanism=mechanism_str,
                            sensitivity=self.dp_params.get('sensitivity', 1.0),
                            gamma=self.dp_params.get('gamma', 0.01),
                            num_blocks=self.sketch_params.get('numBlocks', 1),
                            device=self.device,
                            accountant=self.accountant # Pass the shared accountant object
                        )
                        
                        idx = 0
                        for p in self.local_model.parameters():
                            if p.grad is not None:
                                numel = p.grad.numel()
                                grad_slice = privatized_grad_tensor[idx:idx + numel]
                                p.grad = grad_slice.detach().clone().view_as(p.grad).to(self.device)
                                idx += numel
                    else: # Direct DP application (without Count Sketch)
                        for p in self.local_model.parameters():
                            if p.grad is None: continue
                            if self.dp_type == 'laplace':
                                p.grad = apply_laplace_with_budget(
                                    p.grad,
                                    sensitivity=self.dp_params.get('sensitivity', 1.0),
                                    epsilon=self.dp_params.get('epsilon', 1.0),
                                    gamma=self.dp_params.get('gamma', 1.0),
                                    accountant=self.accountant
                                )
                            elif self.dp_type == 'gaussian':
                                p.grad = apply_gaussian_with_budget(
                                    p.grad,
                                    delta=self.dp_params.get('delta', 1e-5),
                                    epsilon=self.dp_params.get('epsilon', 1.0),
                                    gamma=self.dp_params.get('gamma', 1.0),
                                    accountant=self.accountant
                                )
                            else:
                                raise ValueError(f"Unknown dp_type: {self.dp_type}")

                self.scaler.step(self.optimizer)
                self.scaler.update()
        
        # Return the updated local model parameters
        return self.local_model.state_dict()


class FederatedServer:
    def __init__(self, num_clients: int, total_epsilon: float, total_delta: float, device: torch.device,
                 dp_type: str | None, dp_params: dict, use_count_sketch: bool, sketch_params: dict | None,
                 testloader: torch.utils.data.DataLoader):
        self.num_clients = num_clients
        self.global_model = SimpleCNN().to(device)
        self.accountant = BudgetAccountant(epsilon=total_epsilon, delta=total_delta)
        self.device = device
        self.dp_type = dp_type
        self.dp_params = dp_params
        self.use_count_sketch = use_count_sketch
        self.sketch_params = sketch_params
        self.testloader = testloader
        self.clients: list[FederatedClient] = []

        print(f"Initialized FederatedServer with BudgetAccountant: ε={total_epsilon}, δ={total_delta}")

    def distribute_data_to_clients(self, trainset: torchvision.datasets.CIFAR10, batch_size: int, epochs_per_round: int):
        """Distributes the training data among clients and initializes client objects."""
        data_per_client = len(trainset) // self.num_clients
        
        # Create a list of Subset objects for each client
        client_data_indices = list(range(len(trainset)))
        random.shuffle(client_data_indices) # Shuffle to ensure random distribution

        for i in range(self.num_clients):
            start_idx = i * data_per_client
            end_idx = start_idx + data_per_client
            subset_indices = client_data_indices[start_idx:end_idx]
            client_subset = torch.utils.data.Subset(trainset, subset_indices)
            
            client = FederatedClient(
                client_id=i,
                train_data=client_subset,
                device=self.device,
                dp_type=self.dp_type,
                dp_params=self.dp_params,
                use_count_sketch=self.use_count_sketch,
                sketch_params=self.sketch_params,
                accountant=self.accountant, # Pass the shared accountant
                epochs_per_round=epochs_per_round,
                batch_size=batch_size
            )
            self.clients.append(client)
        print(f"Distributed data to {self.num_clients} clients, each with {data_per_client} samples.")


    def aggregate_models(self, client_model_states: list[dict]) -> dict:
        """
        Aggregates model parameters from clients using Federated Averaging.
        Assumes all clients have the same model architecture.
        """
        if not client_model_states:
            return self.global_model.state_dict()

        # Initialize aggregated state with the first client's model state
        aggregated_state = client_model_states[0].copy()

        # Sum up parameters from all other clients
        for i in range(1, len(client_model_states)):
            for key in aggregated_state:
                aggregated_state[key] += client_model_states[i][key]

        # Average the parameters
        for key in aggregated_state:
            aggregated_state[key] /= len(client_model_states)
            
        return aggregated_state

    def train_federated(self, global_rounds: int):
        """
        Orchestrates the federated learning training process.
        """
        try:
            for round_num in range(global_rounds):
                print(f"\n--- Global Round {round_num + 1}/{global_rounds} ---")

                # 1. Server sends global model to clients
                global_model_state = self.global_model.state_dict()
                for client in self.clients:
                    client.set_global_model(global_model_state)

                # 2. Clients train locally and send updates
                client_updates = []
                for client in self.clients:
                    try:
                        updated_local_model_state = client.train_local()
                        client_updates.append(updated_local_model_state)
                    except BudgetError as be:
                        print(f"Client {client.client_id} stopped due to BudgetError: {be}. Skipping this client for this round.")
                        # Optionally, you could handle this more gracefully, e.g., by not including this client's update
                        # or by stopping the entire training if a critical client runs out of budget.
                        continue
                
                if not client_updates:
                    print("No clients returned updates this round. Stopping federated training.")
                    break

                # 3. Server aggregates updates
                aggregated_state = self.aggregate_models(client_updates)

                # 4. Server updates global model
                self.global_model.load_state_dict(aggregated_state)

                # 5. Evaluate global model
                acc = evaluate(self.global_model, self.testloader)
                eps_used, delta_used = self.accountant.total()
                eps_rem, delta_rem = self.accountant.remaining()

                print(f" Global Round {round_num + 1} Test Accuracy: {acc:.4f}")
                print(f"   Total Used ε: {eps_used:.4f}, δ: {delta_used:.6f}")
                print(f"   Total Remaining ε: {eps_rem:.4f}, δ: {delta_rem:.6f}")

                if eps_rem <= 0 and delta_rem <= 0:
                    print("\nGlobal privacy budget exhausted! Stopping federated training early.")
                    break

        except BudgetError as be:
            print(f"\nFederated training stopped due to BudgetError: {be}")
        except Exception as ex:
            print(f"\nAn unexpected error occurred during federated training: {ex}")

        print("Federated training completed.\n")
        return self.global_model


if __name__ == "__main__":
    total_epsilon = 11000
    total_delta = 1-1e-9 
    global_rounds = 2 # Number of communication rounds between server and clients
    epochs_per_round_client = 3 # Number of local epochs each client runs per global round
    num_federated_clients = 4

    delta = 1e-5
    epsilon = 1.1011632828830176
    gamma = 0.01
    sensitivity = 1.0

    print("===========Parameters for Federated DP Training===========")
    print(f"Running experiments with ε={epsilon}, δ={delta}, γ={gamma}, sensitivity={sensitivity}")
    print(f"Total global rounds: {global_rounds}, local epochs per client: {epochs_per_round_client}")
    print(f"Number of federated clients: {num_federated_clients}")
    print(f"Seed value for reproducibility: {seed}")
    print(f"Batch size: {batch_size}\n")

    # --- Experiment 1: No DP Noise ---
    # print("\n=== Experiment 1: Federated Learning - No DP Noise ===")
    # server_no_dp = FederatedServer(
    #     num_clients=num_federated_clients,
    #     total_epsilon=float('inf'), # Infinite budget for no DP
    #     total_delta=1.0, # Delta is typically 1.0 for no DP
    #     device=device,
    #     dp_type=None,
    #     dp_params={},
    #     use_count_sketch=False,
    #     sketch_params=None,
    #     testloader=testloader
    # )
    # server_no_dp.distribute_data_to_clients(trainset, batch_size, epochs_per_round_client)
    # trained_global_model_no_dp = server_no_dp.train_federated(global_rounds=global_rounds)

    # # --- Experiment 2: Gaussian DP Noise with Budget Accounting ---
    # print("\n=== Experiment 2: Federated Learning - Gaussian DP Noise with Budget Accounting ===")
    # server_gaussian_dp = FederatedServer(
    #     num_clients=num_federated_clients,
    #     total_epsilon=total_epsilon,
    #     total_delta=total_delta,
    #     device=device,
    #     dp_type='gaussian',
    #     dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma, 'sensitivity': sensitivity},
    #     use_count_sketch=False,
    #     sketch_params=None,
    #     testloader=testloader
    # )
    # server_gaussian_dp.distribute_data_to_clients(trainset, batch_size, epochs_per_round_client)
    # trained_global_model_gaussian_dp = server_gaussian_dp.train_federated(global_rounds=global_rounds)

    # # --- Experiment 3: Laplace DP Noise with Budget Accounting ---
    # print("\n=== Experiment 3: Federated Learning - Laplace DP Noise with Budget Accounting ===")
    # server_laplace_dp = FederatedServer(
    #     num_clients=num_federated_clients,
    #     total_epsilon=total_epsilon,
    #     total_delta=0.0, # Delta is typically 0 for pure Laplace
    #     device=device,
    #     dp_type='laplace',
    #     dp_params={'sensitivity': sensitivity, 'epsilon': epsilon, 'gamma': gamma},
    #     use_count_sketch=False,
    #     sketch_params=None,
    #     testloader=testloader
    # )
    # server_laplace_dp.distribute_data_to_clients(trainset, batch_size, epochs_per_round_client)
    # trained_global_model_laplace_dp = server_laplace_dp.train_federated(global_rounds=global_rounds)

    # --- Experiment 4: CSVec + Gaussian DP with Budget Accounting ---
    sketch_rows = 5
    sketch_cols = 10000
    csvec_blocks = 1
    # print(f"\n=== Experiment 4: Federated Learning - CSVec + Gaussian DP with Budget Accounting (r={sketch_rows}, c={sketch_cols}, blocks={csvec_blocks}) ===")
    # server_cs_gaussian = FederatedServer(
    #     num_clients=num_federated_clients,
    #     total_epsilon=total_epsilon,
    #     total_delta=total_delta,
    #     device=device,
    #     dp_type='gaussian',
    #     dp_params={'delta': delta, 'epsilon': epsilon, 'gamma': gamma, 'sensitivity': sensitivity},
    #     use_count_sketch=True,
    #     sketch_params={'d': sketch_rows, 'w': sketch_cols, 'numBlocks': csvec_blocks},
    #     testloader=testloader
    # )
    # server_cs_gaussian.distribute_data_to_clients(trainset, batch_size, epochs_per_round_client)
    # trained_global_model_cs_gaussian = server_cs_gaussian.train_federated(global_rounds=global_rounds)

    # --- Experiment 5: CSVec + Laplace DP with Budget Accounting ---
    print(f"\n=== Experiment 5: Federated Learning - CSVec + Laplace DP with Budget Accounting (r={sketch_rows}, c={sketch_cols}, blocks={csvec_blocks}) ===")
    server_cs_laplace = FederatedServer(
        num_clients=num_federated_clients,
        total_epsilon=total_epsilon,
        total_delta=0.0, # Delta is typically 0 for pure Laplace
        device=device,
        dp_type='laplace',
        dp_params={'delta': delta,'sensitivity': sensitivity, 'epsilon': epsilon, 'gamma': gamma},
        use_count_sketch=True,
        sketch_params={'d': sketch_rows, 'w': sketch_cols, 'numBlocks': csvec_blocks},
        testloader=testloader
    )
    server_cs_laplace.distribute_data_to_clients(trainset, batch_size, epochs_per_round_client)
    trained_global_model_cs_laplace = server_cs_laplace.train_federated(global_rounds=global_rounds)

# -------------OUTPUT-----------------
# Using device: cuda
# Device name: NVIDIA GeForce RTX 3060 Ti
# Running experiments with ε=1.1011632828830176, δ=1e-05, γ=0.01, sensitivity=1.0
# Total global rounds: 2, local epochs per client: 3
# Number of federated clients: 4
# Seed value for reproducibility: 42
# Batch size: 1024


# === Experiment 1: Federated Learning - No DP Noise ===
# Initialized FederatedServer with BudgetAccountant: ε=inf, δ=1.0
# Distributed data to 4 clients, each with 12500 samples.

# --- Global Round 1/2 ---
#  Global Round 1 Test Accuracy: 0.2441
#    Total Used ε: 0.0000, δ: 0.000000
#    Total Remaining ε: inf, δ: 1.000000

# --- Global Round 2/2 ---
#  Global Round 2 Test Accuracy: 0.3428
#    Total Used ε: 0.0000, δ: 0.000000
#    Total Remaining ε: inf, δ: 1.000000
# Federated training completed.


# === Experiment 2: Federated Learning - Gaussian DP Noise with Budget Accounting ===
# Initialized FederatedServer with BudgetAccountant: ε=11000, δ=0.999999999
# Distributed data to 4 clients, each with 12500 samples.

# --- Global Round 1/2 ---
#  Global Round 1 Test Accuracy: 0.2322
#    Total Used ε: 1374.2518, δ: 0.012403
#    Total Remaining ε: 9625.7501, δ: 1.000000

# --- Global Round 2/2 ---
#  Global Round 2 Test Accuracy: 0.3411
#    Total Used ε: 2748.5036, δ: 0.024651
#    Total Remaining ε: 8251.4949, δ: 1.000000
# Federated training completed.


# === Experiment 3: Federated Learning - Laplace DP Noise with Budget Accounting ===
# Initialized FederatedServer with BudgetAccountant: ε=11000, δ=0.0
# Distributed data to 4 clients, each with 12500 samples.

# --- Global Round 1/2 ---
#  Global Round 1 Test Accuracy: 0.2673
#    Total Used ε: 1374.2518, δ: 0.000000
#    Total Remaining ε: 9625.7501, δ: 0.000000

# --- Global Round 2/2 ---
#  Global Round 2 Test Accuracy: 0.3441
#    Total Used ε: 2748.5036, δ: 0.000000
#    Total Remaining ε: 8251.4949, δ: 0.000000
# Federated training completed.

# === Experiment 4: Federated Learning - CSVec + Gaussian DP with Budget Accounting (r=5, c=10000, blocks=1) ===
# Initialized FederatedServer with BudgetAccountant: ε=11000, δ=0.999999999
# Distributed data to 4 clients, each with 12500 samples.

# --- Global Round 1/2 ---
#  Global Round 1 Test Accuracy: 0.2387
#    Total Used ε: 171.7815, δ: 0.001559
#    Total Remaining ε: 10828.2142, δ: 1.000000

# --- Global Round 2/2 ---
#  Global Round 2 Test Accuracy: 0.3486
#    Total Used ε: 343.5629, δ: 0.003115
#    Total Remaining ε: 10656.4336, δ: 1.000000
# Federated training completed.

# === Experiment 5: Federated Learning - CSVec + Laplace DP with Budget Accounting (r=5, c=10000, blocks=1) ===
# Initialized FederatedServer with BudgetAccountant: ε=11000, δ=0.0
# Distributed data to 4 clients, each with 12500 samples.

# --- Global Round 1/2 ---
#  Global Round 1 Test Accuracy: 0.2464
#    Total Used ε: 171.7815, δ: 0.000000
#    Total Remaining ε: 10828.2142, δ: 0.000000

# --- Global Round 2/2 ---
#  Global Round 2 Test Accuracy: 0.3470
#    Total Used ε: 343.5629, δ: 0.000000
#    Total Remaining ε: 10656.4336, δ: 0.000000
# Federated training completed.
