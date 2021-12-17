import numpy as np
import matplotlib.pyplot as plt

# x_initial = np.array([[10], [-1000]])
# x_initial = np.array([[0], [0]])
x_initial = np.array([[5000], [20]])
y = np.array([[0.0], [0.0]])
eps = 10 ** (-10)

def h(x):
    x1 = x[0, 0]
    x2 = x[1, 0]
    ans = np.zeros((2, 1))
    ans[0, 0] = 3 + x1 + 2 * x2 - x1 ** 2 - 2 * x1 * x2
    ans[1, 0] = 4 + 3 * x1 + 4 * x2 - x1 * x2 - 2 * x2 ** 2
    return ans

def dh(x):
    x1 = x[0, 0]
    x2 = x[1, 0]
    ans = np.zeros((2, 2))
    ans[0, 0] = 1 - 2 * x1 - 2 * x2
    ans[1, 0] = 3 - x2
    ans[0, 1] = 2 - 2 * x1
    ans[1, 1] = 4 - x1 - 4 * x2
    return ans

norm_h_list = []
x0_list = []
x1_list = []

x0 = x_initial
d_norm = float('inf')
steps = 0
while d_norm > eps:
    steps += 1
    d = np.linalg.inv(dh(x0)) @ (h(x0) - y)
    d_norm = np.linalg.norm(d)
    x0 = x0 - d
    norm_h_list.append(np.linalg.norm(h(x0)))
    x0_list.append(x0[0, 0])
    x1_list.append(x0[1, 0])
    # print(d_norm)
    

print('------Result-------\n')
print('initial guess = \n', x_initial)
print('h_x* = \n', h(x0))
print('norm_h(x*) = \n', np.linalg.norm(h(x0)))
print('x* = \n', x0)
print('d_norm = ', d_norm)
print('steps = ', steps)


# plt.plot(norm_h_list)
# plt.show()
# plt.plot(x0_list)
# plt.plot(x1_list)
# plt.show()



# x_star = np.array([[-1.2286], [-0.0588]])
# print(h(x_star))
