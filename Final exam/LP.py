from scipy.optimize import linprog

obj = [1, 0, 0]

lhs_ineq = [[-1, -1, 0], 
            [-1, 1, 0], 
            [-1, 0, -1],
            [-1, 0, 1],
            [0, 0.1, 0.1], 
            [0, -0.1, -0.1]]

rhs_ineq = [0, 
            0, 
            0,
            0, 
            -1.9,
            1.9]

opt = linprog(c = obj, A_ub = lhs_ineq, b_ub = rhs_ineq, bounds = [(None, None), (None, None), (None, None)])
print(opt)