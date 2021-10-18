import numpy as np
import scipy.io as sio
import os 
import sys
from os.path import dirname, join as pjoin
import matplotlib.pyplot as plt
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
mat_fname = pjoin(dir_path, 'DataHW07_Prob3.mat')

mat_contents = sio.loadmat(mat_fname)
print("mat_contents include: ", sorted(mat_contents.keys()))

C_pre = mat_contents['C'] # C.shape = (1, 500), C[0][0].shape = (3, 100)
N = mat_contents['N'][0][0] # N = 500
x_actual = mat_contents['x_actual'] # x_actual.shape = (1, 500), x_actual[0][0].shape = (100, 1)
y = mat_contents['y'] # y.shape = (1, 500), y[0][0].shape = (3, 1)

# 3 - (a)
# ------------------------------------------------------
# Rebuild C and Construct A
C = [0] # len(C) = 501 after appending
for i in range(0, N):
    C.append(C_pre[0, i])

C_0 = C[1]
A = [0, C_0] # len(A) = 501. Use A[i] to produce A_i
for i in range(2, N + 1):
    C_0 = np.concatenate((C_0, C[i]), axis=0)
    A.append(C_0)

# n = 34 to guarantee at least 100 independent columns
# print(np.linalg.matrix_rank(A[34]))
n = 34

# 3 - (b)
# ------------------------------------------------------
# Initilize plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Compare different LS runtime')

# Constructing Y
y_0 = y[0, 0]
Y = [0, y_0] # len(A) = 501. Use A[i] to produce A_i
for i in range(1, N):
    y_0 = np.concatenate((y_0, y[0, i]), axis=0)
    Y.append(y_0)

# Constructing R, Use R[i] to produce R_i
R = [0]
for i in range(1, N + 1):
    R.append(np.identity(3 * i))

E = [0] * n
start = time.time()
for k in range(n, N + 1):
    A_k = A[k]
    Y_k = Y[k]
    R_k = R[k]
    x_actual_k = x_actual[0, k - 1]
    x_hat_k = np.linalg.inv(A_k.T.dot(R_k).dot(A_k)).dot(A_k.T).dot(R_k).dot(Y_k)
    x_error_k = x_hat_k - x_actual_k
    norm_k = np.linalg.norm(x_error_k)
    E.append(norm_k)
end = time.time()
print("Run time for calculating norm error using batch Process = ", end - start)

ax1.plot(E[n:], 'r.')
ax1.set(xlabel='k', ylabel='E_k')
ax1.set_title("Norm error in x-hat using Batch Process\n Run time = {:.3f} s".format(end - start))
# plt.show()

# 3 - (c)
# ------------------------------------------------------
# Start from A_34, which is A[34]
A_n = A[n]
Y_n = Y[n]
R_n = R[n]
# Calculate M_34
M_n = np.zeros((100, 100))
S = np.identity(3)
for i in range(1, n + 1):
    C_i = C[i]
    M_n += C_i.T.dot(S).dot(C_i)

E_RLS = [0] * n
E_RLS_IL = [0] * n
X_n = np.linalg.inv(A_n.T.dot(R_n).dot(A_n)).dot(A_n.T).dot(R_n).dot(Y_n)
X_actual_n = x_actual[0, n - 1]
X_error_k = X_n - X_actual_n
norm_k = np.linalg.norm(X_error_k)
# print(norm_k)
E_RLS.append(norm_k)
E_RLS_IL.append(norm_k)


M_0 = M_n
X_0 = X_n
start = time.time()
for i in range(n + 1, N + 1):
    C_1 = C[i]
    Y_1 = y[0][i - 1]
    M_1 = M_0 + C_1.T.dot(S).dot(C_1)
    M_1_inv = np.linalg.inv(M_1)
    X_1 = X_0 + M_1_inv.dot(C_1.T).dot(S).dot(Y_1 - C_1.dot(X_0))
    X_1_actual = x_actual[0, i - 1]
    X_error_1 = X_1 - X_1_actual
    norm = np.linalg.norm(X_error_1)
    E_RLS.append(norm)

    M_0 = M_1
    X_0 = X_1
end = time.time()
print("Run time for calculating norm error using RLS = ", end - start)


ax2.plot(E_RLS[n:], 'b.')
ax2.set(xlabel='k', ylabel='E_k')
ax2.set_title("Norm error in x-hat using RLS\n Run time = {:.3f} s".format(end - start))
# plt.show()



M_0_inv = np.linalg.inv(M_n)
X_0 = X_n
start = time.time()
for i in range(n + 1, N + 1):
    C_1 = C[i]
    Y_1 = y[0][i - 1]
    M_1_inv = M_0_inv - M_0_inv.dot(C_1.T).dot(np.linalg.inv(C_1.dot(M_0_inv).dot(C_1.T) + np.linalg.inv(S)).dot(C_1).dot(M_0_inv))
    X_1 = X_0 + M_1_inv.dot(C_1.T).dot(S).dot(Y_1 - C_1.dot(X_0))
    X_1_actual = x_actual[0, i - 1]
    X_error_1 = X_1 - X_1_actual
    norm = np.linalg.norm(X_error_1)
    E_RLS_IL.append(norm)

    M_0_inv = M_1_inv
    X_0 = X_1
end = time.time()
print("Run time for calculating norm error using RLS with Inversion Lemma = ", end - start)


ax3.plot(E_RLS_IL[n:], 'y.')
ax3.set(xlabel='k', ylabel='E_k')
ax3.set_title("Norm error in x-hat using RLS with Inversion Lemma\n Run time = {:.3f} s".format(end - start))
plt.show()