from scipy.integrate import quad
from scipy.stats import norm, multivariate_normal
import numpy as np


def integrand(x, l1, l2, sigma_y, mu_y):
    return norm().pdf(x) * norm().cdf(-(l2/(l1 * sigma_y)) * x + (l2-mu_y)/sigma_y) - norm().pdf(x) * norm().cdf(-mu_y/sigma_y)


def analytical_calculation_of_the_probability(mu_x, mu_y, sigma_x, sigma_y, l1, l2):
    """
    Test of being in the triangular region
    """
    __b = -l2 * sigma_x / (l1 * sigma_y)
    __a = (l1 * l2 - l1 * mu_y - l2 * mu_x)/(l1 * sigma_y)
    rho = - __b / np.sqrt(1 + __b ** 2)
    rv = multivariate_normal([0., 0.,], [[1.0, rho], [rho, 1.0]])
    cdf_diff = rv.cdf([__a/(np.sqrt(1+__b**2)), (l1 - mu_x)/sigma_x]) - rv.cdf([__a/(np.sqrt(1+__b**2)), (- mu_x)/sigma_x])
    sub_part = norm().cdf(-mu_y/sigma_y) * (norm().cdf((l1 - mu_x)/sigma_x) - norm().cdf((- mu_x)/sigma_x) )
    return cdf_diff - sub_part

a = 2
b = 1

# -----------
# parameter
_c1 = 1.5
_c2 = 2
_l1 = 3
_l2 = 4
_mu_x = 1.5
_var_x = 1.0
_sigma_x = _var_x ** 0.5
_mu_y = 2
_var_y = 0.9
_sigma_y = _var_y ** 0.5
# -----------
I = quad(integrand, -_mu_x/_sigma_x, (_l1 - _mu_x)/_sigma_x, args=(_l1, _l2, _sigma_y, _mu_y))
print(f'One dimensional Integral: {I}')

# This is the same as the result from using formula in the paper, hence, there is some issue about
# >>> cdf_diff - substract
# 0.7872997577245191

# -----------------------------------
# Now, we try to do the integration from beginning, see where the problem comes from
# 注意积分顺序！！！
from scipy.integrate import dblquad
prob = dblquad(lambda x, y: norm(loc=_mu_x, scale=_sigma_x).pdf(x)*norm(loc=_mu_y, scale=_sigma_y).pdf(y), 0, _l2,
               lambda y: 0, lambda y: _l1)
print(f'Total prob in cube: {prob}')

#
from scipy.integrate import dblquad
prob_in_triangle = dblquad(lambda y, x: norm(loc=_mu_x, scale=_sigma_x).pdf(x) * norm(loc=_mu_y, scale=_sigma_y).pdf(y), 0, _l1,
                           lambda x: 0, lambda x: -_l2/_l1 * x + _l2)
print(f'prob in triangle: {prob_in_triangle}')

#
prob_in_transferred_triangle = dblquad(lambda y, x: norm().pdf(x) * norm().pdf(y), -_mu_x/_sigma_x, (_l1-_mu_x)/_sigma_x,
                           lambda x: -_mu_y/_sigma_y, lambda x: (-_l2/_l1 * x + _l2 - _mu_y)/_sigma_y)
print(f'prob in triangle (first wrong transformation): {prob_in_transferred_triangle}')

prob_in_trial_triangle = dblquad(lambda y, x: norm().pdf(x) * norm().pdf(y), -_mu_x/_sigma_x, (_l1-_mu_x)/_sigma_x,
                           lambda x: -_mu_y/_sigma_y, lambda x: ((-_l2/_l1 * (x * _sigma_x + _mu_x) + _l2) - _mu_y)/_sigma_x)
print(f'prob in triangle (trial transformation): {prob_in_trial_triangle}')

print(f'prob in triangle using analytical expression: {analytical_calculation_of_the_probability(_mu_x, _mu_y, _sigma_x, _sigma_y, _l1, _l2)}')
# print(2 * prob[0])
# print('The issue comes from we transform the problem')

# import numpy as np
# prob1 = dblquad(lambda x, y: norm(loc=_mu_x, scale=_sigma_x).pdf(x)*norm(loc=_mu_y, scale=_sigma_y).pdf(y), -np.inf, _l2,
#                lambda y: -np.inf, lambda y: _l1)
# print(f'prob1: {prob1}')