A = [-1 -1 0
    -1 1 0
    -1 0 -1
    -1 0 1
    0 0.1 0.1
    0 -0.1 -0.1]

b = [0
    0
    0
    0
    -1.9
    2.1]

f = [1 0 0]

X_optimized = linprog(f, A, b)

u_star = X_optimized(2:end)

% H = eye(4) * 2
% Aeq =[1 1 0 1; -1 0 1 -1]
% beq = [4; 2]
% f = zeros(4, 1)
% options = optimoptions('quadprog','Display','iter');
% [x,fval] = quadprog(H, f, [], [], Aeq, beq, [], [], [], options)

% A = [1 3; 3 0]
% AA = [0.4 3.2; 2.8 0.4]
% trace(A.' * A)
% trace(AA.' * AA)

% A = [-32.57514 -3.89996 -6.30185 -5.67305 -26.21851;
%      -36.21632 -11.13521 -38.80726 -16.86330 -1.42786;
%      -5.07732 -21.86599 -38.27045 -36.61390 -33.95078;
%      -36.51955 -38.28404 -19.40680 -31.67486 -37.34390;
%      -25.28365 -38.57919 -31.99765 -38.36343 -27.13790];
% 
% [U, Sigma, V] = svd(A)
% 
% d = diag(Sigma);
% d(end) = 0;
% D = diag(d);
% B = U*D*V';
% E = A - B;
% [U, Sigma, V] = svd(E)
% [U, Sigma, V] = svd(B)