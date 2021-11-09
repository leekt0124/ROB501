import numpy as np

A = np.array([[1, 2], [3, 4], [5, 0], [0, 6]], dtype=float)
y = np.array([[1.5377, 3.6948, -7.7193, 7.3621]]).T
Q = np.array([[1, 0.5, 0.5, 0.25], [0.5, 2, 0.25, 1], [0.5, 0.25, 2, 1], [0.25, 1, 1, 4]], dtype=float)
P = np.array([[0.5, 0.25], [0.25, 0.5]])

def MVE(A, Q, y, i):
    A_ = A[:i, :]
    Q_ = Q[:i, :i]
    y_ = y[:i, :]
    K =  np.linalg.inv(A_.T @ np.linalg.inv(Q_) @ A_ + np.linalg.inv(P)) @ A_.T @ np.linalg.inv(Q_)
    print(f"x_hat for first {i} row = \n", K @ y_)
    print(f"covariance of x_hat for first {i} row = \n", P - P @ A_.T @ np.linalg.inv(A_ @ P @ A_.T + Q_) @ A_ @ P)
    # print("check", np.linalg.inv(A_.T @ np.linalg.inv(Q_) @ A_))


MVE(A, Q, y, 1)
MVE(A, Q, y, 2)
MVE(A, Q, y, 3)
MVE(A, Q, y, 4)

