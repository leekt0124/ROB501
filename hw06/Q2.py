import numpy as np
import scipy.io as sio
import os 
import sys
from os.path import dirname, join as pjoin
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
mat_fname = pjoin(dir_path, 'DataHW06_Prob2.mat')

mat_contents = sio.loadmat(mat_fname)
print("mat_contents include: ", sorted(mat_contents.keys()))

# examine shape of data
dy = mat_contents['dy']
t = mat_contents['t']
y = mat_contents['y']
y_test = np.sin(2 * np.pi * t)
n = dy.shape[0]
print("shape of dy = ", dy.shape, ", ", "shape of t = ", t.shape, ", ", "shape of y = ", y.shape, "n = ", n)

# Compute derivative
dy_naive = np.zeros_like(dy)
for i in range(1, t.shape[0]):
    dy_naive[i] = (y[i] - y[i - 1]) / (t[i] - t[i - 1])


# A = np.column_stack(t**2, t)
# A = np.column_stack((np.ones_like(t), t, t ** 2, t ** 3, t ** 4, t ** 5, t ** 6, t ** 7))
# A = np.column_stack((np.ones_like(t), np.sin(np.pi * t), np.sin(2 * np.pi * t), ))
# A = np.ones_like(t)
# print(A.shape)
# for i in range(1, 5):
#     A = np.column_stack((A, np.sin(i * np.pi * t)))
# alpha_hat = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(y)
# alpha_hat = np.squeeze(alpha_hat)
# alpha_hat = [-2.36972306e-04,  5.52418525e-01, -8.70354829e-01,  1.25668888e+00, 2.21206197e-01]

# Build windows
WINDOW_SIZE = 30
NUM_DIVISION = n // WINDOW_SIZE

# A = np.ones((WINDOW_SIZE, 1))
# for i in range(1, 5):
#     A = np.column_stack((A, np.sin(i * np.pi * t)))
alpha_hat_list = []
for i in range(NUM_DIVISION):
# for i in range(1):
    # for j in range(WINDOW_SIZE * i, min(WINDOW_SIZE * (i + 1), n)):
    dy_window = dy[WINDOW_SIZE * i: min(WINDOW_SIZE * (i + 1), n)]
    t_window = t[WINDOW_SIZE * i: min(WINDOW_SIZE * (i + 1), n)]
    y_window = y[WINDOW_SIZE * i: min(WINDOW_SIZE * (i + 1), n)]

    A = np.ones_like(t_window)
    dA = np.zeros_like(t_window)
    for j in range(1, 3):
        A = np.column_stack((A, t_window ** j))
        dA = np.column_stack((dA, j * t_window ** (j - 1)))

    # A = np.sin(np.pi * t_window)
    # dA = np.pi * np.cos(np.pi * t_window)
    # A = np.column_stack((A, np.cos(np.pi * t_window)))
    # dA = np.column_stack((A, -np.pi * np.sin(np.pi * t_window)))

    # for i in range(2, 4):
    #     A = np.column_stack((A, np.sin(i * np.pi * t_window)))
    #     dA = np.column_stack((dA, i * np.pi * np.cos(i * np.pi * t_window)))
    alpha_hat = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(y_window)
    alpha_hat_list.append(alpha_hat)
    # plt.plot(A[:, 4])
    # plt.plot(dA[:, 4])
    # plt.show()
    # print(alpha_hat)
    # if i == 0:
    #     print(A)
    #     print(dA)
    #     print(alpha_hat)


def poly_fit(alpha_list, t):
    func = np.ones_like(t)
    d_func = np.ones_like(t)
    for i in range(NUM_DIVISION):
        t_window = t[WINDOW_SIZE * i: min(WINDOW_SIZE * (i + 1), n)]
        y_fit = 0
        dy_fit = 0
        alpha = alpha_list[i]

        # for j in range(len(alpha)):
        #     y_fit += alpha[j] * np.sin((j + 1) * np.pi * t_window)
        #     dy_fit += alpha[j] * (j + 1) * np.pi * np.cos((j + 1) * np.pi * t_window)

        # y_fit = alpha[0] * np.sin(np.pi * t_window) + alpha[1] * np.cos(np.pi * t_window)
        # dy_fit = alpha[0] * np.pi * np.cos(np.pi * t_window) + alpha[1] * (-np.pi) * np.sin(np.pi * t_window)
        # func[WINDOW_SIZE * i: min(WINDOW_SIZE * (i + 1), n)] = y_fit
        # d_func[WINDOW_SIZE * i: min(WINDOW_SIZE * (i + 1), n)] = dy_fit

        for j in range(len(alpha)):
            y_fit += alpha[j] * t_window ** j
            if j != 0:
                dy_fit += alpha[j] * j * t_window ** (j - 1)
        func[WINDOW_SIZE * i: min(WINDOW_SIZE * (i + 1), n)] = y_fit
        d_func[WINDOW_SIZE * i: min(WINDOW_SIZE * (i + 1), n)] = dy_fit
    return func, d_func

# print(poly_fit(alpha_hat, t))


# Plot for 2.(a)
plt.plot(t, dy, 'r', label='dy')
plt.plot(t, dy_naive, label='dy_naive')
# plt.plot(t, y_test, label='y_test')
# fit, d_fit = sin_fit(alpha_hat_list, t)
# ax1.plot(t, fit, 'g', label='sin_fit')
# ax2.plot(t, d_fit, 'g', label='d_sin_fit')
plt.xlabel("t")
plt.ylabel("function")
plt.title("dy - t (groundtruth vs naive fitting)")
plt.legend(loc = 'upper right')
# fig.show()
plt.show()

# Plot for 2.(b)
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('fit for y and dy using moving window with size of %s' % (WINDOW_SIZE))
# plot of data
ax1.plot(t, y, 'r', label='y')
ax2.plot(t, dy, 'r', label='dy')
# plt.plot(t, dy_naive, label='dy_naive')
# plt.plot(t, y_test, label='y_test')
fit, d_fit = poly_fit(alpha_hat_list, t)
ax1.plot(t, fit, 'g', label='poly_fit')
ax2.plot(t, d_fit, 'g', label='d_poly_fit')

ax1.set_xlabel("t")
ax2.set_xlabel("t")
ax1.set_title("y - t")
ax2.set_title("dy - t")
# plt.xlabel("t")
# plt.ylabel("function")
# plt.title("function of y and dy")
ax1.legend(loc = 'upper right')
ax2.legend(loc = 'upper right')
# fig.show()
plt.show()