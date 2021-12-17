function Segway_anim(t,phi,theta,Ts);
%SEGWAY_ANIM    Inverted pendulum animation.
%    SEGWAY_ANIM(T,X,PHI,TS) is an animation of the segway platform
%
%      T - time vector
%      PHI - pendulum angle vector (with respect to vertical)
%      THETA - angle of the wheel w.r.t. the pendulum 
%      TS - an optional animation scaling factor with default
%          .1; increase to speed animation
%
%    Run SEWAY_ANIM with no arguments for a demo.
%
%Jessy Grzzle, based on the file
% pend_anim by Eric Westervelt
%23July2009
%
%X - cart position vector (green wedge is zero)
%

if nargin < 4, Ts = .1; end
if nargin == 0, t = 0:.1:6*pi; x = 5*sin(t); phi = .5*cos(t); end

% define constants
scl = 1;    % scaling for better visualization
cl  = 8;    % length of cart
ch  = 3;    % height of cart
br  = 0.5;  % radius of pendulum
pl  = 10;   % pendulum length
wr = pl/5;     % wheel radius

% check arguments
x= (phi+theta)*wr;
[nt,mt] = size(t); [nx,mx] = size(x); [nh,mh] = size(phi); [ntheta,mtheta] = size(theta);

if ((nt ~=1) & (mt ~=1)),
    error('first argument must all be a vector');
end
if ((nx ~=1) & (mx ~=1))
    error('second argument must all be a vector');
end
if ((nh ~=1) & (mh ~=1))
    error('third argument must all be a vector');
end

if nt < mt, t = t'; end
if nx < mx, x = x'; end
if nh < mh, phi = phi'; end
if ntheta < mtheta, theta = theta'; end

[t,tmp] = even_sample(t, [x phi theta], 1/Ts);
x = tmp(:,1);
phi = tmp(:,2);
theta = tmp(:,3);



% set up figure window
xmax   = 20;
ymax   = 20;
ydepth = 10;
figure(100); clf
axis([-xmax xmax -ydepth ymax]); axis off

arrow = line([0 -1],[0 -1]);
set(arrow,'LineWidth',2,'Color','g');
arrow = line([0 1],[0 -1]);
set(arrow,'LineWidth',2,'Color','g');

ground = line([-xmax xmax],[0 0]);
set(ground,'Color',[0 0 0],'LineWidth',2);

x0 = 0;
xcart = [x0-cl/2 x0+cl/2 x0+cl/2 x0-cl/2];
ycart = [ch ch 0 0];
cartcolor = 'b';
cart = patch(xcart,ycart,cartcolor);

phi0 = 3/4*pi;
xpendulum = [x0 x0+(pl-br)*sin(phi0)];
ypendulum = [ch ch+(pl-br)*cos(phi0)];
pendulum = line(xpendulum,ypendulum);
set(pendulum,'Color',[0 0 0],'LineWidth',6);
param = linspace(0,2*pi+2*pi/50,50);
xbob = br*cos(param);
ybob = br*sin(param);
bob = patch(xbob+pl*sin(phi0),ybob+pl*cos(phi0),'r');

% Wheel rim and hub
%
s = linspace(0,2*pi,100);
rim_coord = wr*[cos(s); sin(s)];
hub_coord = .15*wr*[cos(s); sin(s)];

% Build up wheel spokes
%
flag='y'; % yes for all spokes.
%
R = inline('[cos(th) -sin(th); sin(th) cos(th)]'); % rotation matrix
spoke_coord = wr*[-0.05 0.05 0.1 -0.1;
    0.0  0.0  1.0  1.0];
spokes = [];
key_spoke_coord = R(-pi/2)*spoke_coord;
for k = -1:5
    spokes = [spokes R(k*pi/4)*spoke_coord];
end
spoke_coord = spokes;

key_spoke = patch(key_spoke_coord(1,:),key_spoke_coord(2,:),'r');
set(key_spoke,'EdgeColor','r','LineWidth',1);

% determined above whether or not to plot all spokes
if strcmp(upper(flag),'Y')
    spokes = patch(spoke_coord(1,:),spoke_coord(2,:),'k');
    set(spokes,'LineWidth',1);
end

rim = patch(x(1)+rim_coord(1,:),rim_coord(2,:),'k');
set(rim,'FaceColor','none','LineWidth',5);
hub = patch(x(1)+hub_coord(1,:),hub_coord(2,:),'k');
set(hub,'LineWidth',2);




% animate the system
for k = 1:length(t)
    % refresh cart
    x1 = scl*x(k);
    xcart = [-cl/2 cl/2 cl/2 -cl/2];
    ycart = [ch ch 0 0];
%     z=R(phi(k))*[xcart;ycart];
     z=R(-phi(k))*[xcart;ycart];
    
    set(cart,'XData',x1+z(1,:),'YData',wr+z(2,:));

    % refresh pendulum
    xpendulum = [x(k) x(k)+(pl-br)*sin(phi(k))];
    ypendulum = wr+[0  (pl-br)*cos(phi(k))];
    set(pendulum,'XData',xpendulum,'YData',ypendulum);
    set(bob,'XData',x(k)+xbob+pl*sin(phi(k)),...
        'YData',wr+ybob+pl*cos(phi(k)));

    key_spoke_coord_tmp = R(-theta(k))*key_spoke_coord;
    set(key_spoke,'XData',x(k)+key_spoke_coord_tmp(1,:),...
        'YData',wr+key_spoke_coord_tmp(2,:));

    % determined above whether or not to plot all spokes
    if strcmp(upper(flag),'Y')
        spoke_coord_tmp = R(-theta(k))*spoke_coord;
        set(spokes,'XData',x(k)+spoke_coord_tmp(1,:),...
            'YData',wr+spoke_coord_tmp(2,:));
    end


    set(rim,'XData',x(k)+rim_coord(1,:),...
        'YData',wr+rim_coord(2,:));
    set(hub,'XData',x(k)+hub_coord(1,:),...
        'YData',wr+hub_coord(2,:));
T=floor(100*t(k))/100;
title(['Time = ',num2str(T)],'fontsize',18)
    drawnow;
    pause(.01);
end

%---------------------------------------------------------------

function [Et, Ex] = even_sample(t, x, Fs);

% Obtain the process related parameters
N = size(x, 2);    % number of signals to be interpolated
M = size(t, 1);    % Number of samples provided
t0 = t(1,1);       % Initial time
tf = t(M,1);       % Final time
EM = (tf-t0)*Fs;   % Number of samples in the evenly sampled case with
% the specified sampling frequency
Et = linspace(t0, tf, EM)';

% Using cubic spline interpolation technique interpolate and
% re-sample each signal and obtain the evenly sampled signal.
for s = 1:N,
    Ex(:,s) = spline(t(:,1), x(:,s), Et(:,1));
end;