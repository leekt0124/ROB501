import numpy as np

# Q6 - (a)
def L(M):
    Q = np.array([[0.9, 0.2], [-0.1, 1]])
    K = np.array([[0.9, 0.1], [0.2, 1]])
    return Q.dot(M).dot(K) - np.transpose(M)

v1 = np.array([[1, 0], [0, 0]])
v2 = np.array([[0, 0], [0, 1]])
v3 = np.array([[0, 1], [0, 0]])
v4 = np.array([[0, 0], [1, 0]])
print(L(v1))
print(L(v2))
print(L(v3))
print(L(v4))

# Q7 - (d)
G = np.array([[3, 0, 1], [0, 2, -1], [1, -1, 2]])
print(np.linalg.inv(G))
beta = np.array([[3], [-1], [2]])
print(np.linalg.inv(G).dot(beta))

