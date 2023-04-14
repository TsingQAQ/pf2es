import numpy as np
# https://stats.stackexchange.com/questions/162560/distribution-of-the-largest-fragment-of-a-broken-stick-spacings


def exact_cdf_calc(x, k=49):
    pass


def gumbel_cdf_approax(x, k=49):
    return np.exp(-np.exp(- (k + 1) * x + np.log(k + 1)))


from matplotlib import pyplot as plt

xs = np.linspace(0, 1, 100000)
plt.plot(xs, gumbel_cdf_approax(xs))
plt.title('Gumbel approximation of maximum spacing for population size 50')
plt.ylabel('probability')
plt.show()