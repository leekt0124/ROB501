

load SegwayDataTest


phi=zeros(N,1);
theta=zeros(N,1);
xk=x0;
tic
t=zeros(1,N);
for k =1:N
    uk=u(k);
    xkp1=A*xk+B*uk;  % x[k+1] = A x[k] + B u[k];
    phi(k)=[1 0 0 0]*xkp1;
    theta(k)=[0 1 0 0]*xkp1;
    t(k)=k*Ts;
    xk=xkp1;
end
toc
Segway_anim(t,phi,theta,Ts);

