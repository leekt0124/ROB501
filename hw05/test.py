import numpy as np

A = np.array([[1, 0, 0, 0, 0], [0, 0.5, 0, 0, 0], [0, 0, 0.5, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0.5]])
B = np.array([1, 0, 2, 0, 3]).T
C = 0.2
D = B.T

print(np.linalg.inv(A + B*C*D))