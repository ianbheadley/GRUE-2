import numpy as np
from scipy.stats import skew, kurtosis, norm, entropy

def bimodality_coefficient(x):
    # standard formula for sample bimodality coefficient
    # (skewness^2 + 1) / kurtosis (where kurtosis is regular, not excess)
    # n denominator adjustment is ignored for large N (10000)
    g = skew(x)
    k = kurtosis(x, fisher=False) 
    return (g**2 + 1) / k if k != 0 else 0

def kl_divergence_gaussian(x, bins=100):
    # compute histograms
    hist, bin_edges = np.histogram(x, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # gaussian dist with same mu, std
    mu, std = x.mean(), x.std()
    q = norm.pdf(bin_centers, loc=mu, scale=std)
    
    # filter out zeros
    hist = hist + 1e-10
    q = q + 1e-10
    
    # KL divergence sum P * log(P/Q) * dx
    dx = bin_edges[1] - bin_edges[0]
    return np.sum(hist * np.log(hist / q)) * dx

# Test
x = np.random.randn(10000)
print("Normal:", bimodality_coefficient(x), kl_divergence_gaussian(x))
y = np.concatenate([np.random.randn(5000)-3, np.random.randn(5000)+3])
print("Bimodal:", bimodality_coefficient(y), kl_divergence_gaussian(y))
