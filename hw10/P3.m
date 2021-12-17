A = 1;
B = 0.1;
u = 10;
R = 16;
c = 3 * (10 ^ 8);
C = -2 / c;
Q = 10 ^ (-18);

z_1 = 2.2 * (10 ^ (-8));

X_0 = 1;
P_0 = 0.25;
X_hat = A * X_0 + B * u;

z_hat = 2 / c * (5 - X_hat);

P_hat = A * P_0 * A' + B * R * B';
K = P_hat * C' / (C * P_hat * C' + Q);
X_1_hat = X_hat + K * (z_1 - z_hat)
P_1_hat = P_hat -  K * C * P_hat
