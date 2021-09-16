%% First Example, Diagonal Covaraince Matrix
N=500;
mu = [0 0];
Sigma = [10 0; 0 1]; R = chol(Sigma);
X = repmat(mu,N,1) + randn(N,2)*R;
figure(1)
plot(X(:,1), X(:,2),'*')
axis('equal')
xlabel('X_1'), ylabel('X_2')


%% Second Example, Non-diagonal Covariance Matrix
mu = [0 0];
O=[1 1; -1 1]/sqrt(2);
Sigma = O*[10 0; 0 1]*O'; R = chol(Sigma);
X = repmat(mu,N,1) + randn(N,2)*R;
figure(2)
plot(X(:,1), X(:,2),'*')
axis('equal')
xlabel('X_1'), ylabel('X_2')


%% Third Example, Non-diagonal Covariance Matrix, but do transformation on the variables
mu = [0 0];
O=[1 1; -1 1]/sqrt(2);
Sigma = O*[10 0; 0 1]*O'; R = chol(Sigma);
X = repmat(mu,N,1) + randn(N,2)*R;
Y=O*X';
%Y=O'*X';
figure(3)
plot(Y(1,:), Y(2,:),'*')
axis('equal')
xlabel('Y_1'), ylabel('Y_2')