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
n = dy.shape[0]
dt = t[1] - t[0]
print("shape of dy = ", dy.shape, ", ", "shape of t = ", t.shape, ", ", "shape of y = ", y.shape, "n = ", n)

# Compute derivative
dy_naive = np.zeros_like(dy)
for i in range(1, t.shape[0]):
    dy_naive[i] = (y[i] - y[i - 1]) / dt

# Build windows
WINDOW_SIZE = 30
NUM_DIVISION = n // WINDOW_SIZE
POLY_POWER = 4
alpha_hat_list = []

for i in range(NUM_DIVISION):
    dy_window = dy[WINDOW_SIZE * i: min(WINDOW_SIZE * (i + 1), n)]
    t_window = t[WINDOW_SIZE * i: min(WINDOW_SIZE * (i + 1), n)]
    y_window = y[WINDOW_SIZE * i: min(WINDOW_SIZE * (i + 1), n)]

    A = np.ones_like(t_window)
    dA = np.zeros_like(t_window)
    for j in range(1, POLY_POWER + 1):
        A = np.column_stack((A, t_window ** j))
        dA = np.column_stack((dA, j * t_window ** (j - 1)))
    
    alpha_hat = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(y_window)
    # alpha_hat = np.linalg.pinv(A).dot(y_window)
    alpha_hat_list.append(alpha_hat)



def poly_fit(alpha_list, t):
    func = np.ones_like(t)
    d_func = np.ones_like(t)
    for i in range(NUM_DIVISION):
        t_window = t[WINDOW_SIZE * i: min(WINDOW_SIZE * (i + 1), n)]
        y_fit = 0
        dy_fit = 0
        alpha = alpha_list[i]

        for j in range(len(alpha)):
            y_fit += alpha[j] * t_window ** j
            if j != 0:
                dy_fit += alpha[j] * j * t_window ** (j - 1)
        func[WINDOW_SIZE * i: min(WINDOW_SIZE * (i + 1), n)] = y_fit
        d_func[WINDOW_SIZE * i: min(WINDOW_SIZE * (i + 1), n)] = dy_fit
    return func, d_func

# Plot for 2.(a)
plt.plot(t, dy_naive, 'y', label='dy_naive')
plt.plot(t, dy, 'c', linewidth=5, label='dy')
fit, d_fit = poly_fit(alpha_hat_list, t)
plt.plot(t, d_fit, 'r', label='d_poly_fit')

plt.xlabel("t")
plt.ylabel("function")
plt.title("deriviative of y (groundtruth vs fitting)")
plt.legend(loc = 'upper right')

plt.show()

# Compute RMSE
def calRMSE(n, dy, dy_estimated):
    return np.sqrt(np.sum((dy - dy_estimated) ** 2) / n)

print("RSME = ", calRMSE(n, dy, d_fit))