import numpy as np
import scipy.io as sio
import os 
import sys
from os.path import dirname, join as pjoin
import matplotlib.pyplot as plt
import time
import collections

dir_path = os.path.dirname(os.path.realpath(__file__))
mat_fname = pjoin(dir_path, 'DataHW07_Prob4.mat')

mat_contents = sio.loadmat(mat_fname)
print("mat_contents include: ", sorted(mat_contents.keys()))

C_pre = mat_contents['C'] # C.shape = (1, 500), C[0][0].shape = (3, 20)
N = mat_contents['N'][0][0] # N = 500
x_actual = mat_contents['x_actual'] # x_actual.shape = (1, 500), x_actual[0][0].shape = (20, 1)
y = mat_contents['y'] # y.shape = (1, 500), y[0][0].shape = (3, 1)

# Initilize plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Drifting state estimation')

# 4 - (a)
# ------------------------------------------------------
# Rebuild C and Construct A
C = [0] # len(C) = 501 after appending
for i in range(0, 500):
    C.append(C_pre[0, i])

C_0 = C[1]
A = [0, C_0] # len(A) = 501. Use A[i] to produce A_i
for i in range(2, 501):
    C_0 = np.concatenate((C_0, C[i]), axis=0)
    A.append(C_0)

# n = 7 to guarantee at least 20 independent columns
# print(np.linalg.matrix_rank(A[7]))
n = 7

# 4 - (b)
# Constructing Y
y_0 = y[0, 0]
Y = [0, y_0] # len(Y) = 501. Use Y[i] to produce Y_i
for i in range(1, 500):
    y_0 = np.concatenate((y_0, y[0, i]), axis=0)
    Y.append(y_0)

R_origin = [0]
for i in range(1, N + 1):
    R_origin.append(np.identity(3 * i))

E = [0] * n
start = time.time()
for k in range(n, N + 1):
    A_k = A[k]
    Y_k = Y[k]
    R_k = R_origin[k]
    x_actual_k = x_actual[0, k - 1]
    x_hat_k = np.linalg.inv(A_k.T.dot(R_k).dot(A_k)).dot(A_k.T).dot(R_k).dot(Y_k)
    x_hat_k = np.linalg.inv(A_k.T.dot(A_k)).dot(A_k.T).dot(Y_k)
    x_error_k = x_hat_k - x_actual_k
    norm_k = np.linalg.norm(x_error_k)
    E.append(norm_k)
end = time.time()
print("Run time for calculating norm error using batch Process = ", end - start)

ax1.plot(E[n:], 'r.')
ax1.set(xlabel='k', ylabel='E_k')
ax1.set_title("Norm error in x-hat using original Batch Process\n Run time = {:.3f} s".format(end - start))


# # Constructing R_forget, Use R_forget[i] to produce R_forget_i
# FORGET_FACTOR = 0.98
# R_0 = np.identity(3)
# R_forget = [0, R_0]
# for i in range(1, 501):
#     # R.append(np.identity(3 * i))
#     R_new = R_0.copy()
#     R_new *= FORGET_FACTOR
#     size_ = R_new.shape[0]
#     R_new = np.concatenate((R_new, np.zeros((size_, 3))), axis=1)
#     R_new = np.concatenate((R_new, np.zeros((3, size_ + 3))), axis=0)
#     R_new[-3:, -3:] = np.identity(3)
#     R_forget.append(R_new)
#     R_0 = R_new


FORGET_FACTOR = 0.98
R_forget = [0]
R_0 = np.identity(3)
d = collections.deque()

for i in range(1, 501):
    for j in range(3):
        d.appendleft(FORGET_FACTOR ** (i - 1))
    R_forget.append(np.diag(d))


E_forget = [0] * n
start = time.time()
for k in range(n, N + 1):
    A_k = A[k]
    Y_k = Y[k]
    R_k = R_forget[k]
    x_actual_k = x_actual[0, k - 1]
    x_hat_k = np.linalg.inv(A_k.T.dot(R_k).dot(A_k)).dot(A_k.T).dot(R_k).dot(Y_k)
    x_error_k = x_hat_k - x_actual_k
    norm_k = np.linalg.norm(x_error_k)
    E_forget.append(norm_k)
end = time.time()
print("Run time for calculating norm error using batch Process = ", end - start)

ax2.plot(E_forget[n:], 'b.')
ax2.set(xlabel='k', ylabel='E_k')
ax2.set_title("Norm error in x-hat using Batch Process\n Forgetting factor = {}\n Run time = {:.3f} s".format(FORGET_FACTOR, end - start))



# Start from A_7, which is A[7]
A_n = A[n]
Y_n = Y[n]
R_n = R_forget[n]
# Calculate M_34
M_n = np.zeros((20, 20))
G_n = np.zeros((20, 1))
for i in range(1, n + 1):
    C_i = C[i]
    M_n += FORGET_FACTOR ** (n - i) * C_i.T.dot(C_i)
    G_n += FORGET_FACTOR ** (n - i) * C_i.T.dot(y[0, i - 1])



E_RLS = [0] * n
E_RLS_IL = [0] * n
X_n = np.linalg.inv(M_n).dot(G_n)
X_actual_n = x_actual[0, n - 1]
X_error_k = X_n - X_actual_n
norm_k = np.linalg.norm(X_error_k)

E_RLS.append(norm_k)
E_RLS_IL.append(norm_k)


# M_0 = M_n
# X_0 = X_n
# start = time.time()
# for i in range(n + 1, N + 1):
#     C_1 = C[i]
#     # print(C_1[0, :5])
#     Y_1 = y[0][i - 1]
#     # print(Y_1[:5, 0])
#     M_1 = FORGET_FACTOR * M_0 + C_1.T.dot(C_1)
#     K_1 = np.linalg.inv(M_1).dot(C_1.T)
#     X_1 = X_0 + K_1.dot(Y_1 - C_1.dot(X_0))
#     # print(X_1[:10])
#     X_1_actual = x_actual[0, i - 1]
#     # print(X_1_actual[:10])
#     X_error_1 = X_1 - X_1_actual
#     norm = np.linalg.norm(X_error_1)
#     # print(norm)
#     E_RLS.append(norm)

#     M_0 = M_1
#     X_0 = X_1

# end = time.time()
# print("Run time for calculating E_k using RLS = ", end - start)

# ax3.plot(E_RLS[n:], 'r.')
# ax3.set(xlabel='k', ylabel='E_K')
# ax3.set_title("Norm error in x-hat using RLS\n Forgetting factor = {}\n Run time = {:.3f} s".format(FORGET_FACTOR, end - start))
# plt.show()


M_0_inv = np.linalg.inv(M_n)
X_0 = X_n
start = time.time()
for i in range(n + 1, N + 1):
    C_1 = C[i]
    Y_1 = y[0][i - 1]
    M_1_inv = 1 / FORGET_FACTOR * M_0_inv - 1 / FORGET_FACTOR * M_0_inv.dot(C_1.T).dot(np.linalg.inv(FORGET_FACTOR * np.identity(3) + C_1.dot(M_0_inv).dot(C_1.T))).dot(C_1).dot(M_0_inv)
    X_1 = X_0 + M_1_inv.dot(C_1.T).dot(Y_1 - C_1.dot(X_0))

    X_1_actual = x_actual[0, i - 1]
    X_error_1 = X_1 - X_1_actual
    norm = np.linalg.norm(X_error_1)
    E_RLS_IL.append(norm)    

    M_0_inv = M_1_inv
    X_0 = X_1

end = time.time()
print("Run time for calculating norm error using RLS with Inversion Lemma = ", end - start)

ax3.plot(E_RLS_IL[n:], 'y.')
ax3.set(xlabel='k', ylabel='E_K')
ax3.set_title("Norm error in x-hat using RLS with Inversion Lemma\n Forgetting factor = {}\n Run time = {:.3f} s".format(FORGET_FACTOR, end - start))
plt.show()


# 4 - (bonus)
# ------------------------------------------------------

for FORGET_FACTOR in np.linspace(1, 0.25, 4):
    # E_RLS = [0] * n
    E_RLS_IL = [0] * n
    X_n = np.linalg.inv(M_n).dot(G_n)
    X_actual_n = x_actual[0, n - 1]
    X_error_k = X_n - X_actual_n
    norm_k = np.linalg.norm(X_error_k)

    E_RLS_IL.append(norm_k)

    M_0_inv = np.linalg.inv(M_n)
    X_0 = X_n
    start = time.time()
    for i in range(n + 1, N + 1):
        C_1 = C[i]
        Y_1 = y[0][i - 1]
        M_1_inv = 1 / FORGET_FACTOR * M_0_inv - 1 / FORGET_FACTOR * M_0_inv.dot(C_1.T).dot(np.linalg.inv(FORGET_FACTOR * np.identity(3) + C_1.dot(M_0_inv).dot(C_1.T))).dot(C_1).dot(M_0_inv)
        X_1 = X_0 + M_1_inv.dot(C_1.T).dot(Y_1 - C_1.dot(X_0))

        X_1_actual = x_actual[0, i - 1]
        X_error_1 = X_1 - X_1_actual
        norm = np.linalg.norm(X_error_1)
        E_RLS_IL.append(norm)    

        M_0_inv = M_1_inv
        X_0 = X_1

    plt.plot(E_RLS_IL[n:], label='FORGET_FACTOR = {}'.format(FORGET_FACTOR))

plt.xlabel("k")
plt.ylabel("E_k")
plt.title("Norm error for different forget factors")
plt.legend(loc = 'upper right')
plt.show()