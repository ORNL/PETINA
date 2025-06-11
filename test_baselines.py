from petina import baselines
import numpy as np

#domain = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
domain = [1,2,3,4,5,1,2,3,4,5]
#domain = torch.tensor([[1, 2, 3], [4, 5, 6]])


sensitivity = 1
epsilon = 0.1
delta = 10e-5
gamma = 0.001


print(baselines.applyFlipCoin(probability=0.9, items=[1,2,3,4,5,6,7,8,9,10]))  
print("DP = ", baselines.applyDPLaplace(domain, sensitivity, epsilon))
print("DP = ", baselines.applyDPGaussian(domain, delta, epsilon))
print("DP = ", baselines.applyDPExponential(domain, sensitivity, epsilon))
print("Percentile Privacy=", baselines.percentilePrivacy(domain, 10))  
print("unary encoding = ", baselines.unaryEncoding(domain, p=.75, q=.25))   # can add p and q value or can use default p and q
print("histogram encoding 1 = ", baselines.histogramEncoding(domain))
print("histogram encoding 2 = ", baselines.histogramEncoding_t(domain))
print("clipping = ", baselines.applyClippingDP(domain, 0.4, 1.0, 0.1))
print("adaptive clipping = ", baselines.applyClippingAdaptive(domain))
print("pruning = ", baselines.applyPruning(domain, 0.8))
print("adaptive pruninig = ", baselines.applyPruningAdaptive(domain))
print("pruning+DP = ", baselines.applyPruningDP(domain, 0.8, sensitivity, epsilon))
print(baselines.get_p(epsilon))
print(baselines.get_q(p=0.5,eps=epsilon))
print(baselines.get_gamma_sigma(p=0.5,eps=epsilon))#
print(baselines.above_threshold_SVT(.3, domain, T=.5, epsilon=epsilon))
