load SegwayData4KF.mat

phi = zeros(N, 1);
theta = zeros(N, 1);
phi_dot = zeros(N, 1);
theta_dot = zeros(N, 1);
K_1 = zeros(N, 1);
K_2 = zeros(N, 1);
K_3 = zeros(N, 1);
K_4 = zeros(N, 1);

x1 = x0;
P1 = P0;
tic
t=zeros(1,N);
for k =1:N
    uk = u(k);
    yk = y(k);

    K = P1 * C' / (C * P1 * C' + Q)
    x1 = A * x1 + B * uk + A * K * (yk - C * x1);
    P1 = A * (P1 - K * C * P1) * A' + G * R * G';
        
    x1_hat = x1;
    P1_hat = P1;

    phi(k)=[1 0 0 0] * x1_hat;
    theta(k)=[0 1 0 0] * x1_hat;
    phi_dot(k) = [0 0 1 0] * x1_hat;
    theta_dot(k) = [0 0 0 1] * x1_hat;
    K_1(k) = K(1);
    K_2(k) = K(2);
    K_3(k) = K(3);
    K_4(k) = K(4);

    t(k)=k*Ts;
    x1=x1_hat;
    P1 = P1_hat;
end
toc
% Segway_anim(t,phi,theta,Ts);
% subplot(2, 1, 1);
% plot(phi);
% subplot(2, 1, 2);
% plot(theta);

plot(t, phi, t, theta);
title("\phi, \theta - t");
legend("\phi", "\theta");


figure

plot(t, phi_dot, t, theta_dot);
title("\Phi\_dot, \theta\_dot - t");
legend("\Phi\_dot", "\theta\_dot");

figure

plot(t, K_1, t, K_2, t, K_3, t, K_4);
legend("K1", "K2", "K3", "K4");
title("K_k - t")
% Segway_anim(t,phi,theta,Ts);

[Kss, Pss] = dlqe(A, G, C, R, Q)


%     x1 = A * x1 + B * uk;  % x[k+1] = A x[k] + B u[k];
%     P1 = A * P1 * A' + G * R * G'
%     K = P1 * C' / (C * P1 * C' + Q)
%     x1_hat = x1 + K * (yk - C * x1);
%     P1_hat = P1 - K * C * P1;

