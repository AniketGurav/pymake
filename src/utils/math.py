import numpy as np
import scipy as sp


def lognormalize(x):
    a = np.logaddexp.reduce(x)
    return np.exp(x - a)

def categorical(params):
    return np.where(np.random.multinomial(1, params) == 1)[0]

def bernoulli(param, size=1):
    return np.random.binomial(1, param, size=size)

### Power law distribution generator
def random_powerlaw(alpha, x_min, size=1):
    ### Discrete
    alpha = float(alpha)
    u = np.random.random(size)
    x = (x_min-0.5)*(1-u)**(-1/(alpha-1))+0.5
    return np.floor(x)

### A stick breakink process, truncated at K components.
def gem(gmma, K):
    sb = np.empty(K)
    cut = np.random.beta(1, gmma, size=K)
    for k in range(K):
        sb[k] = cut[k] * cut[0:k].prod()
    return sb


