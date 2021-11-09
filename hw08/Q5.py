import numpy as np

A = np.array([[1, 2], [3, 4], [5, 0], [0, 6]], dtype=float)
y = np.array([[1.5377, 3.6948, -7.7193, 7.3621]]).T
# Q = np.array([[1, 0.5, 0.5, 0.25], [0.5, 2, 0.25, 1], [0.5, 0.25, 2, 1], [0.25, 1, 1, 4]], dtype=float)
Q = np.identity(4)
# P = np.array([[0.5, 0.25], [0.25, 0.5]])
P = np.identity(2) * 10 ** 6

def LS(A, y):
    print("LS = ", np.linalg.inv(A.T @ A) @ A.T @ y)

def BLUE(A_, Q_, y_):
    K =  np.linalg.inv(A_.T @ np.linalg.inv(Q_) @ A_) @ A_.T @ np.linalg.inv(Q_)
    print("BLUE = ", K @ y_)
    # print(f"covariance of x_hat for first {i} row = \n", K @ Q_ @ K.T)

def MVE(A_, Q_, y_):
    K =  np.linalg.inv(A_.T @ np.linalg.inv(Q_) @ A_ + np.linalg.inv(P)) @ A_.T @ np.linalg.inv(Q_)
    print("MVE = ", K @ y_)
    # print(f"covariance of x_hat for first {i} row = \n", P - P @ A_.T @ np.linalg.inv(A_ @ P @ A_.T + Q_) @ A_ @ P)
    # print("check", np.linalg.inv(A_.T @ np.linalg.inv(Q_) @ A_))

# def Q6():
#     x_bar = np.array([[1], [-1]])
#     y_bar = A @ x_bar
#     x_hat = x_bar + 
#     print()

LS(A, y)
BLUE(A, Q, y)
MVE(A, Q, y)



