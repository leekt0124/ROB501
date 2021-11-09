import numpy as np

C = np.array([[1, 2], [3, 4], [5, 0], [0, 6]], dtype=float)
y = np.array([[1.5377, 3.6948, -7.7193, 7.3621]]).T
Q = np.array([[1, 0.5, 0.5, 0.25], [0.5, 2, 0.25, 1], [0.5, 0.25, 2, 1], [0.25, 1, 1, 4]], dtype=float)
# Q = np.identity(4)
P = np.array([[0.5, 0.25], [0.25, 0.5]])
# P = np.identity(2) * 10 ** 6

def Q6():
    x_bar = np.array([[1], [-1]])
    y_bar = C @ x_bar
    x_hat = x_bar + P @ C.T @ np.linalg.inv(C @ P @ C.T + Q) @ (y - y_bar)
    print(x_hat)

Q6()


# Q2
S = np.array([[4, 2], [2, 2]])
S_ = np.array([[4, 2, 0], [2, 2, 0], [0, 0, 5]])
print(np.linalg.inv(S))
print(np.linalg.inv(S_))

