function [D,C,G,B,J,R,F_bar,M_bar]= dyn_mod_segway(q,dq)
%DYN_MOD_SEGWAY

%24-Jul-2009 10:49:36


%
% Authors(s): Grizzle
%
%
% Model NOTATION: Spong and Vidyasagar, page 142, Eq. (6.3.12)
%                 D(q)ddq + C(q,dq)*dq + G(q) = B*tau
%
%
%
%
%
%
phi=q(1);theta=q(2);
dphi=dq(1);dtheta=dq(2);
%
%
%
%
D=zeros(2,2);
D(1,1)=7*cos(phi)+146/15;
D(1,2)=7/2*cos(phi)+11/3;
D(2,1)=7/2*cos(phi)+11/3;
D(2,2)=11/3;
%
%
%
C=zeros(2,2);
C(1,1)=-7/2*sin(phi)*dphi;
C(2,1)=-7/2*sin(phi)*dphi;
%
%
%
%
G=zeros(2,1);
G(1)=-7*sin(phi);
G(2)=0;
%
%
%
%
B=zeros(2,1);
B(2,1)=1;
%
%
%
%
%
% From Reyhanoglu et al., Dynamics and Control of a Class of Underactuated Mechanical Systems
% IEEE Transaction Auto Control, Vol. 44, No. 9, September, 1999, pp. 1663-1671
%
M_bar=D([2:2],[2:2])-D([2:2],[1])*inv(D(1,1))*D([1],[2:2]);
F=C*dq+G; F_bar=F([2:2])-D([2:2],[1])*inv(D(1,1))*F(1);
%
J=-D(1,2)/D(1,1);
R=-F(1)/D(1,1);
%
%u=F_bar+M_bar*v;
%
%
return