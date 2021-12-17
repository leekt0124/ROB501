% symb_model_planar_segway.m
%
clear *
%
% This is for EECS 560 and 562: Planar Segway model
%
%
% J = inertia of body about CoM; Jw = effective inertia of wheel = wheel +
%                  motor inertia plus gearing
%
% m = mass of the pendulum and M = mass of the "cart" which means
%                        wheel + motor and is assumed at the axel
%
% reference: phi = absolute angle for the body, measured clockwise from the vertical
%
% reference: theta = relative angle of the wheel w.r.t. the body, measured
%    clockwise from the body ===> absolute angle of wheel = body angle +  wheel angle
%
% reference:  x = position measured from left to right = R * absolute angle
% of the wheel
%
% L = distance to center of mass of the pendulum
%
% reference: y = vertical height; set zero at radius of wheel = R.


% R*(theta + phi) = x  ==> theta = x/R  - phi

if 0 % general model
    syms g L R m M J Jw
else % special parameters for EECS 560
    g=1; % g=1 means that time has been normalized.
    M=3*g;
    m=7*g;
    L=1;
    R=L/2;
    J=m*L^2/6;
    Jw=J;    
end

if 0 % coordinate choices
    syms phi dphi x dx
    q=[phi x].';
    dq=[dphi dx].';
    theta = x/R - phi
    dtheta = dx/R - dphi
else
    syms phi dphi theta dtheta
    q=[phi theta].';
    dq=[dphi dtheta].';
    x = R*(theta + phi)
    dx = R*(dtheta + dphi)
end



%
%
pcart=[x;0];
ppend_centermass=pcart+L*[sin(phi);cos(phi)];
pcm = (M * pcart + m * ppend_centermass)/(m+M)
%
%
% vcart =  jacobian(pcart,q)*dq;
% vpend_centermass = jacobian(ppend_centermass,q)*dq;
vcm = jacobian(pcm,q)*dq;
%
%
KEcm = simplify((1/2)*(M+m)*vcm.'*vcm)

KErotation=simplify(J*(1/2)*(dphi)^2 + (1/2)*Jw*(dtheta + dphi)^2)
%
%
KE = (KEcm+KErotation);
KE = simple(KE)
%
%
%
PE = g*(m+M)*pcm(2);
PE = simple(PE);
%
%
% Model NOTATION: Spong and Vidyasagar, page 142, Eq. (6.3.12)
%                 D(q)ddq + C(q,dq)*dq + G(q) = B*tau
%
L=KE-PE;

G=jacobian(PE,q).';
G=simple(G);
D=simple(jacobian(KE,dq).');
D=simple(jacobian(D,dq));

syms C real
n=max(size(q));
for k=1:n
    for j=1:n
        C(k,j)=0*g;
        for i=1:n
            C(k,j)=C(k,j)+(1/2)*(diff(D(k,j),q(i))+diff(D(k,i),q(j))-diff(D(i,j),q(k)))*dq(i);
        end
    end
end
C=simple(C);
%
% Compute the matrix for the input force
%
E=theta;
B=jacobian(E,q).'
%
%
% Compute D^{-1}
%
d=simple(det(D));
DI=inv(D);
DI=simple(DI);
%
%
% compute RHS of model in the form ddq = f + g mu
f=DI*(-C*dq-G); f=simple(f)
g=DI*B; g=simple(g)
pretty(f(1))
pretty(f(2))
pretty(g(1))
pretty(g(2))
save work_symb_model_planar_segway


% Compute terms in Normal Form

% From Reyhanoglu et al., Dynamics and Control of a Class of Underactuated Mechanical Systems
%%IEEE Transaction Auto Control, Vol. 44, No. 9, September, 1999, pp. 1663-1671
%
M_bar=D([2:2],[2:2])-D([2:2],[1])*inv(D(1,1))*D([1],[2:2]);
F=C*dq+G; F_bar=F([2:2])-D([2:2],[1])*inv(D(1,1))*F(1);
%
J1=-D(1,2)/D(1,1); J1=simple(J1)
R=-F(1)/D(1,1); R=simple(R)
mdVdq0=-jacobian(PE,q(1));
%
%u=F_bar+M_bar*v;

% Compute terms in linear model

D_e=subs(D,'phi','0'); D_e=vpa(D_e,5)
grad_G = jacobian(G,q); grad_G = subs(grad_G,'phi','0'); grad_G = vpa(grad_G,5)

A21=-inv(D_e)*grad_G
B2=inv(D_e)*B



%
save work_symb_model_planar_segway
%
generate_spong_modified_model_2DOF
return

