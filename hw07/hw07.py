import numpy as np

# Q1
# A = np.array([[2, -1, 0], [-1, 2, 1], [0, 1, 2]])
# w, v = np.linalg.eig(A)
# sigma = np.diag(w)
# print(v.dot(sigma).dot(v.T))
# print(A)
# print(w)
# print(v)

# Q2
# A = np.array([[1, 3], [3, 9]])
# w, v = np.linalg.eig(A)
# sigma = np.diag(w)
# print(np.sqrt(sigma))
# print(w, v)
# N = v.dot(np.sqrt(sigma)).dot(v.T)
# print(N.dot(N.T))
# print(N)

# def Q2(A):
#     w, v = np.linalg.eig(A)
#     sigma = np.diag(w)
#     N = v.dot(np.sqrt(sigma)).dot(v.T)
#     print("e-values (diagonal elements of matrix sigma)= \n", w)
#     print("e-vectors (matrix O)= \n", v)
#     print("squared root (N) = \n", N)
#     print("N ^ 2 = \n", N.dot(N.T))
#     print("v * v.T = \n", v.dot(v.T))
#     print("v.T * A * v = \n", v.T.dot(A).dot(v))
# # A = np.array([[1, 3], [3, 9]])
# # A = np.array([[6, 10, 11], [10, 19, 19], [11, 19, 21]])
# # A = np.array([[2, 6, 10], [6, 10, 14], [10, 14, 18]])
# # A = np.array([[1, 0, 6], [0, 4, 7], [6, 7, 10]])
# # A = np.array([[1, 2, 6], [2, 5, 7], [6, 7, 60]])
# # A = np.array([[1, 0, 2 ** 0.5], [0, 2, 0], [2 ** 0.5, 0, 0]])
# # A = np.array([[-1, 0], [0, 2]])
# A = np.array([[0, -0.5774, 0.8165], [2, 0, 0], [0, 0.8165, 0.5774]])
# K = np.array([[1, 0, 2 ** 0.5], [0, 2, 0], [2 ** 0.5, 0, 0]])

# Q2(A)
# print(A.T.dot(K).dot(A))



# # B = np.array([[6], [7]])
# # A = np.array([[1, 2], [2, 5]])
# # print(B)
# # print(A)
# # print(B.T.dot(np.linalg.inv(A)).dot(B))


# # Q7
# # A = np.array([[1, 3, 2], [3, 8, 4]])
# # b = np.array([[1], [2]])
# # x_hat = A.T.dot(np.linalg.inv(A.dot(A.T)).dot(b))
# # print(x_hat)
# # print(A.dot(x_hat))

# Q = np.array([[5, 1, 9], [1, 2, 1], [9, 1, 17]])
# A = np.array([[1, 3, 2], [3, 8, 4]])
# b = np.array([[1], [2]])
# x_hat_weighted = np.linalg.inv(Q).dot(A.T).dot(np.linalg.inv(A.dot(np.linalg.inv(Q).dot(A.T)))).dot(b)
# print(x_hat_weighted)


# A = np.array([[1, 2, 3]])
# A = np.arange(1, 26).reshape(5, 5)
# print(A)
# # print(A[-3:, -3:])
# A[-3:, -3:] = np.identity(3)
# print(A)

# print(A[:, 1])
# print(A[0, 1])

# d = [i for i in range(1, 10)]
# factor = 0.9
# d = []
# for i in range(2, -1, -1):
#     for j in range(3):
#         d.append(factor ** i)
# print(d)
# A = np.diag(d)
# print(A)

import collections
FORGET_FACTOR = 0.98
R_forget = [0]
R_0 = np.identity(3)
d = collections.deque()
diag_list = []

for i in range(1, 5):
    for j in range(3):
        d.appendleft(FORGET_FACTOR ** (i - 1))
    diag_list.append(np.diag(d))
    print(np.diag(d))
# print(diag_list)