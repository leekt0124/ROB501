import numpy as np
import scipy.io as sio
import os 
import sys
from os.path import dirname, join as pjoin
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
mat_fname = pjoin(dir_path, 'DataHW06_Prob3.mat')

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

# Build windows
WINDOW_SIZE = 30
NUM_DIVISION = n // WINDOW_SIZE

alpha_hat_list = []
for i in range(NUM_DIVISION):
    dy_window = dy[WINDOW_SIZE * i: min(WINDOW_SIZE * (i + 1), n)]
    t_window = t[WINDOW_SIZE * i: min(WINDOW_SIZE * (i + 1), n)]
    y_window = y[WINDOW_SIZE * i: min(WINDOW_SIZE * (i + 1), n)]
    # A = np.sin(np.pi * t_window)
    # dA = np.pi * np.cos(np.pi * t_window)
    # for i in range(2, 4):
    #     A = np.column_stack((A, np.sin(i * np.pi * t_window)))
    #     dA = np.column_stack((dA, i * np.pi * np.cos(i * np.pi * t_window)))

    A = np.ones_like(t_window)
    dA = np.zeros_like(t_window)
    for j in range(1, 4):
        A = np.column_stack((A, t_window ** j))
        dA = np.column_stack((dA, j * t_window ** (j - 1)))
    
    # A = np.sin(np.pi * t_window)
    # dA = np.pi * np.cos(np.pi * t_window)
    # A = np.column_stack((A, np.cos(np.pi * t_window)))
    # dA = np.column_stack((A, -np.pi * np.sin(np.pi * t_window)))

    alpha_hat = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(y_window)
    # alpha_hat = np.linalg.pinv(A).dot(y_window)
    alpha_hat_list.append(alpha_hat)



def sin_fit(alpha_list, t):
    func = np.ones_like(t)
    d_func = np.ones_like(t)
    for i in range(NUM_DIVISION):
        t_window = t[WINDOW_SIZE * i: min(WINDOW_SIZE * (i + 1), n)]
        y_fit = 0
        dy_fit = 0
        alpha = alpha_list[i]
        # for j in range(len(alpha) // 2):
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

# Plot for 2.(a)
plt.plot(t, dy_naive, 'y', label='dy_naive')
plt.plot(t, dy, 'r', label='dy_ground_truth')
fit, d_fit = sin_fit(alpha_hat_list, t)
plt.plot(t, d_fit, 'g', label='dy_poly_fit')

plt.xlabel("t")
# plt.ylabel("function")
plt.title("dy - t (naive vs polynomial fit)")
plt.legend(loc = 'upper right')

plt.show()

# Compute RMSE
def calRMSE(n, dy, dy_estimated):
    # print(n)
    # print(dy.shape)
    # print(dy_estimated.shape)
    return np.sqrt(np.sum((dy - dy_estimated) ** 2) / n)
print(calRMSE(n, dy, d_fit))