import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Our 2-dimensional distribution will be over variables X and Y
N = 1000
X = np.linspace(-2, 2, N)
Y = np.linspace(-2, 2, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([0., 0.])
Sigma = np.array([[ 1. , 0.5], [0.5,  1.]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)
Z[np.logical_or(X>0.5,Y>0.5)] = 0

# plot using subplots
fig, ax2 = plt.subplots()
ax2.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis)

plt.hlines(0.5, -2, 2, linewidth=2, colors='r')
plt.vlines(0.5, -2, 2, linewidth=2, colors='r')

plt.title(r'Conditional Distribution  $q(h_{X} \vert \mathcal{F})$  ' +
          '\n  $\mathcal{F} = 0.5, M=1, d=1, C=0, q=2$', fontsize=25)
plt.xticks([])
plt.yticks([])
plt.text(0.9, -1, '$h_{x_1} \succ \mathcal{F}$' + '\n$h_{x_2} \prec \mathcal{F}$', color='w', fontsize=25)
plt.text(-1, 0.9, '$h_{x_2} \succ \mathcal{F}$'+ '\n$h_{x_1} \prec \mathcal{F}$', color='w', fontsize=25)
plt.text(0.9, 0.9, '$h_{x_2} \succ \mathcal{F}$'+ '\n$h_{x_1} \succ \mathcal{F}$',color='w',  fontsize=25)
plt.text(-1.9, -1.9, '$\overline{A}_{X}$', color='w', fontsize=25)
plt.scatter(0.5, 0.5, color='w', s=50, zorder=40)
plt.xlabel(' Batch Element $h_{x_1}$', fontsize=25)
plt.ylabel(' Batch Element $h_{x_2}$', fontsize=25)
plt.tight_layout()
# plt.show()
plt.savefig('Demonstration of Batch', dpi=300)