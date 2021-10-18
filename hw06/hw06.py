import numpy as np
A = np.array([[5, 3], [3, 4]])
b = np.array([[4, 1]]).T
print(A)
print(b)
print(np.linalg.inv(A).dot(b))

y1 = np.array([[1, 0], [2, 0]])
y2 = np.array([[1, 1], [1, 1]])
X = np.array([[0, -1], [2, 0]])
X_hat = 1 / 11 * np.array([[6, -7], [19, -7]])
print(((X - X_hat).T).dot(y1))
print(((X - X_hat).T).dot(y2))


def matrixInverseLemma(A_inv, B, C, D):
    return A_inv - A_inv.dot(B).dot(np.linalg.inv(np.linalg.inv(C) + D.dot(A_inv).dot(B))).dot(D).dot(A_inv)

def verify(A, B, C, D):
    return np.linalg.inv(A + 0.2 * B.dot(D))

A = np.array([[1, 0, 0, 0, 0], [0, 0.5, 0, 0, 0], [0, 0, 0.5, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0.5]])
A_inv = np.linalg.inv(A)
B = np.array([[1, 0, 2, 0, 3]]).T
C = np.array([[0.2]])
D = B.T
print(matrixInverseLemma(A_inv, B, C, D))
print(verify(A, B, C, D))