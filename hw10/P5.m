A = [4.041, 7.046, 3.014; 10.045, 17.032, 7.027; 16.006, 27.005, 11.048];
[U, S, V] = svd(A);
d = diag(S);
d(end) = 0
D = diag(d);
B = U * D * V'
E = B - A
max(sqrt(eig(E' * E)))
[U, S, V] = svd(E)