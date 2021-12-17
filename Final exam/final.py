import numpy as np

# # Q5
# # (b)
# print(np.array([[1, 2], [2, 5]]) - np.array([[0], [2 ** 0.5]]) @ np.array([[0, 2 ** 0.5]]))

# # (c)
# A = np.array([[4, 1], [1, 5]])
# print(np.linalg.inv(A))

# print(np.linalg.eig(np.array([[1, 2], [2, 3]])))
print(np.linalg.eig(np.array([[1, 0, 2], [0, 4, 1], [2, 1, 5]])))


# =================================================
# # Q6
# # (a)
# y0 = np.array([[2], [4], [0]])
# C0 = np.array([[1, 1], [0, 2], [1, -1]])
# mu0 = np.array([[0], [0], [0]])
# Q0 = np.array([[0.7, -0.4, -0.1], [-0.4, 0.8, 0.2], [-0.1, 0.2, 0.3]])

# x_hat = np.linalg.inv(C0.T @ np.linalg.inv(Q0) @ C0) @  C0.T @ np.linalg.inv(Q0) @ y0
# print(x_hat)

# # # (b)
# # Recursive method
# x0 = np.array([[1.1875], [1.5625]])
# C1 = np.array([[1, 2]])
# y1 = 4
# mu1 = 0
# Q1 = np.array([[0.01]])
# M0 = C0.T @ np.linalg.inv(Q0) @ C0
# M1 = M0 + C1.T @ np.linalg.inv(Q1) @ C1
# x1 =  x0 + np.linalg.inv(M1) @ C1.T @ np.linalg.inv(Q1) @ (y1 - C1 @ x0)
# print("M0 = ", M0)
# print("M1 = ", M1)
# print("x1 = ", x1)


# # # batch method
# # y0 = np.array([[2], [4], [0], [4]])
# # C0 = np.array([[1, 1], [0, 2], [1, -1], [1, 2]])
# # mu0 = np.array([[0], [0], [0], [0]])
# # Q0 = np.array([[0.7, -0.4, -0.1, 0], [-0.4, 0.8, 0.2, 0], [-0.1, 0.2, 0.3, 0], [0, 0, 0, 0.01]])

# # x_hat = np.linalg.inv(C0.T @ np.linalg.inv(Q0) @ C0) @  C0.T @ np.linalg.inv(Q0) @ y0
# # print(x_hat) # [[1.04052098] [1.48335745]]


# =================================================
# Q7
A = np.array([[1, 0.1], [0, 1]])
R = np.array([[0.02, 0.04], [0.04, 0.08]])
C1 = np.array([[1, 1]])
C2 = np.array([[0.1, 1]])

z2 = np.array([[2], [0.5]])
P2 = np.array([[2, 1], [1, 4]])
Q1 = 2
Q2 = 1

z3_2 = A @ z2
print(z3_2)
P3_2 = A @ P2 @ A.T + R
print("P3_2 = ", P3_2)

K3_1 = P3_2 @ C1.T @ np.linalg.inv(C1 @ P3_2 @ C1.T + Q1)
print("K3_1 = ", K3_1)
y3_1 = 2.5
z3_3_1 = z3_2 + K3_1 @ (y3_1 - C1 @ z3_2)
print("z3_3_1 = ", z3_3_1)


K3_2 = P3_2 @ C2.T @ np.linalg.inv(C2 @ P3_2 @ C2.T + Q2)
print("K3_2 = ", K3_2)
y3_2 = 0.7
z3_3_2 = z3_2 + K3_2 @ (y3_2 - C2 @ z3_2)
print("z3_3_2 = ", z3_3_2)

P3_3_1 = P3_2 - K3_1 @ C1 @ P3_2
print("P3_3_1 = ", P3_3_1)
print("norm of P3_3_1 = ", np.linalg.norm(P3_3_1))
# print((1.04 ** 2 + 2 * 0.3803 ** 2 + 1.3643 ** 2) ** 0.5)

P3_3_2 = P3_2 - K3_2 @ C2 @ P3_2
print("P3_3_2 = ", P3_3_2)
print("norm of P3_3_2 = ", np.linalg.norm(P3_3_2))


