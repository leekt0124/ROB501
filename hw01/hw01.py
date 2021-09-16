import numpy as np
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

from scipy import integrate

# # Q3 - (b)
# M = np.array([[2, 1], [1, 3]])
# w, v = np.linalg.eig(M)
# print('e-values = ', w)
# print('e-vectors = ', v)
# # print(5/2 + 5**0.5 / 2, 5/2 - 5**0.5 / 2)
# print(np.inner(v[0], v[1]))
# print(np.matmul(v[0].reshape(1, 2), v[1].reshape(2, 1)).squeeze())
# # print(v[0].reshape(1, 2).shape)

# # Q3 - (d)
# m, n = 3, 5
# for _ in range(2):
#     A = np.random.rand(n, m)
#     print('A = ', A)
#     M = np.matmul(A.T, A)
#     print('M = ', M)
#     w, v = np.linalg.eig(M)
#     print('w = ', w)
#     print('v = ', v)
#     selections = random.sample(range(0, 3), 2)
#     print('selections = ', selections)
#     print('we select: ', v[selections[0]], v[selections[1]])
#     print('inner product of e-vectors = ', np.inner(v[selections[0]], v[selections[1]]))
#     print('sum of e-values = ', sum(w))
#     print('trace of the matrix = ', np.trace(M))
#     print('multiplication of e-values = ', np.prod(w))
#     print('det of the matrix = ', np.linalg.det(M))

# # Q4 - (a)
# mu = 0
# variance_1 = 1
# variance_2 = 3 
# sigma_1 = math.sqrt(variance_1)
# sigma_2 = math.sqrt(variance_2)
# x_1 = np.linspace(mu - 3*sigma_1, mu + 3*sigma_1, 100)
# x_2 = np.linspace(mu - 3*sigma_2, mu + 3*sigma_2, 100)
# plt.plot(x_1, stats.norm.pdf(x_1, mu, sigma_1), 'g', label='std = 1')
# plt.plot(x_2, stats.norm.pdf(x_2, mu, sigma_2), 'r', label='std = 3')
# plt.legend()
# plt.title('Gaussian random variable plot')
# plt.show()

# # Q4 - (b)
# mu = 2
# sigma = 5
# f = lambda x: stats.norm.pdf(x, mu, sigma)
# # f = lambda x: 1 / (sigma * (2 * np.pi) ** 0.5) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
# print('area [4, inf]: ', integrate.quad(f, 4.0, np.inf)[0])
# print('area [-2, 4]: ', integrate.quad(f, -2.0, 4.0)[0])
# print('area [-2, 4] or [8, 100]: ', integrate.quad(f, -2, 4)[0] + integrate.quad(f, 8, 100)[0])

# # Q7 - (d)
# mean = [1, 2]
# cov = [[3, 5 ** 0.5], [5 ** 0.5, 2]]  # diagonal covariance
# n_samples = 5000
# x, y = np.random.multivariate_normal(mean, cov, n_samples).T
# plt.plot(x, y, 'x')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title(f'Distribution of x, y with {n_samples} samples')
# plt.axis('equal')
# plt.show()


# # Calculate theoratical value
# mu_x = mean[0]
# mu_y = mean[1]
# sigma_x = cov[0][0] ** 0.5
# sigma_y = cov[1][1] ** 0.5
# rho = cov[0][1] / sigma_x / sigma_y
# y = 10
# mu_conditional = mu_x + rho * sigma_x / sigma_y * (y - mu_y)
# sigma_coditional = (1 - rho ** 2) ** 0.5 * sigma_x
# print('mu_conditional = ', mu_conditional)
# print('sigma_coditional = ', sigma_coditional)

# x = np.linspace(mu_conditional - 3 * sigma_coditional, mu_conditional + 3 * sigma_coditional, 100) # A nparray
# # y = 10
# # f = lambda x, y: (1 / 2 / np.pi * np.exp(-3 * ((x - 1) ** 2 / 3 - 2 * (5/6) ** 0.5 * (x - 1) / 3 ** 0.5 * (y - 2) / 2 ** 0.5 + (y - 2) ** 2 / 2))) / (1 / (4 * np.pi) ** 0.5 * np.exp(-(y - 2) ** 2 / 4))
# f = lambda x: (1/ (2 * np.pi) ** 0.5 / sigma_coditional) * np.exp(- (x - mu_conditional) ** 2 / (2 * sigma_coditional ** 2))
# # f = lambda x, y: (1 / 2 / np.pi * np.exp(-3 * ((x - 1) ** 2 / 3)))
# # f = lambda x, y: x + y
# # print(f(x, y))
# plt.plot(x, f(x), 'r')
# plt.title('Conditional density of X given Y is 10')
# plt.show()
