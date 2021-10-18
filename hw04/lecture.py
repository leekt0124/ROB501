import numpy as np

A = np.array([[1, 0, 0, 0], [0, 1.5, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
print(np.linalg.inv(A))

B = np.array([[1, 1, 2, 1, 3], [2, 0, 8, 1, 3], [-1, 0, -4, 1, 0], [3, 2, 8, 1, 6]])
# B = np.array([[1, 1, 1], [1, 0, 2], [1, 0, -1], [1, 2, 3]])
print(np.linalg.matrix_rank(B))

C = np.array([[1, 1, 1], [1, 2, 2], [1, 2, 3]])
D = np.linalg.inv(C)
print(D)
print(D.dot(np.array([[8], [7], [4]])))

# Q3
P = np.array([[1, 1, 1], [1, 2, 2], [1, 2, 3]])
print(np.linalg.inv(P))