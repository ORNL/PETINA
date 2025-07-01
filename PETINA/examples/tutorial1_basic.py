from petina import algorithms
import numpy as np

#domain = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
# domain = [1,2,3,4,5,1,2,3,4,5]
#domain = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Generate 10 synthetic "like" numbers between 10 and 1000
domain = [random.randint(10, 1000) * random.choice(domain) for _ in range(10)]

print("Synthetic like numbers:", domain)
sensitivity = 1
epsilon = 0.1
delta = 10e-5
gamma = 0.001


print(algorithms.applyFlipCoin(probability=0.9, items=[1,2,3,4,5,6,7,8,9,10]))  
print("DP = ", algorithms.applyDPLaplace(domain, sensitivity, epsilon))
print("DP = ", algorithms.applyDPGaussian(domain, delta, epsilon))
print("DP = ", algorithms.applyDPExponential(domain, sensitivity, epsilon))
print("Percentile Privacy=", algorithms.percentilePrivacy(domain, 10))  
print("unary encoding = ", algorithms.unaryEncoding(domain, p=.75, q=.25))   # can add p and q value or can use default p and q
print("histogram encoding 1 = ", algorithms.histogramEncoding(domain))
print("histogram encoding 2 = ", algorithms.histogramEncoding_t(domain))
print("clipping = ", algorithms.applyClippingDP(domain, 0.4, 1.0, 0.1))
print("adaptive clipping = ", algorithms.applyClippingAdaptive(domain))
print("pruning = ", algorithms.applyPruning(domain, 0.8))
print("adaptive pruninig = ", algorithms.applyPruningAdaptive(domain))
print("pruning+DP = ", algorithms.applyPruningDP(domain, 0.8, sensitivity, epsilon))
print(algorithms.get_p(epsilon))
print(algorithms.get_q(p=0.5,eps=epsilon))
print(algorithms.get_gamma_sigma(p=0.5,eps=epsilon))#
print(algorithms.above_threshold_SVT(.3, domain, T=.5, epsilon=epsilon))
