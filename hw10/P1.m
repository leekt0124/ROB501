% % (a) and (b)
% x1 = 1
% x2 = 3
% x3 = -1
% d = 0.001
% x3 = x3 + d
% f1 = 3 * x1 * (2*x2 - x3 ^ 3) + (x2 ^ 4) / 3
% x3 = x3 - 2 * d
% f2 = 3 * x1 * (2*x2 - x3 ^ 3) + (x2 ^ 4) / 3
% (f1 - f2) / 2 / d

% (c)
x0 = [1, 1, 1, 1, 1];
k = 1:10;
A = ones(1, 10);
B = (10 .^ -k);
Derivative = [];
for axis = 1:5
    result = [];
    for i = 1: 10
        delta = B(i);
        x0(axis) = x0(axis) + delta;
        f_r = funcPartC(x0);
        x0(axis) = x0(axis) - 2 * delta;
        f_l = funcPartC(x0);
        d = (f_r - f_l) / (2 * delta);
        result = [result, d];
        x0(axis) = x0(axis) + delta;
    end
    Derivative = [Derivative, d];

    subplot(2, 3, axis)
    plot(result, 'r-')
    title("Derivative of " + axis + "axis")
    xlabel('step size = 10 \^ (-x)')
    ylabel('Derivative')
end

Derivative
