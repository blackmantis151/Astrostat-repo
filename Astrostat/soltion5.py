import random
import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import norm
import scipy.optimize as opt
###################################################################################
def chi2_func(params):
    mu, sigma = params
    expected_probs = norm.cdf(edges[1:], mu, sigma) - norm.cdf(edges[:-1], mu, sigma)
    expected_counts = expected_probs * len(nums)
    mask = expected_counts > 0  # avoid division by zero
    return np.sum((counts[mask] - expected_counts[mask])**2 / expected_counts[mask])
####################################################################
mu1=0
sigma1 = 1
nums = [random.gauss(mu1, sigma1) for _ in range(100)]
nums = np.array(nums)
#####################################################################

bin_edges = np.arange(min(nums), max(nums) + 0.05, 0.05)
counts, edges = np.histogram(nums, bins=bin_edges)
#########################################################################

result = opt.minimize(chi2_func, [0, 1], bounds=[(-5, 5), (0.1, 5)])
best_mu, best_sigma = result.x

print("Best fit μ:", best_mu)
print("Best fit σ:", best_sigma)


plt.hist(nums, bins=bin_edges, density=True, alpha=0.6, label="Observed")
x = np.linspace(min(nums), max(nums), 500)
plt.plot(x, norm.pdf(x, best_mu, best_sigma), 'r-', label="Fitted Normal PDF")
plt.legend()
plt.show()
		