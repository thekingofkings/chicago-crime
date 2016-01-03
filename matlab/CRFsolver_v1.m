%{

Version 1 of the CRF-based crime rate model.

min_{alpha, w} ||X alpha − y||_1 + ||F w − y_p ||_1

%}


F = importdata('F.csv');
y = importdata('Y.csv');
Yp = importdata('Yp.csv');
X = importdata('X.csv');



% calculat the first half

alpha = ones(size(X, 2), 1);
z = X * alpha - y;
rho = 1;
theta = ones(size(z));


s = (X' * X) \ X';


cnt = 0;
while (true)
    
    % update alpha
    alpha = s * (z + y + theta);

    u = X * alpha - y - theta;
    lambda = 1 / rho;
    x = zeros( size(z) );
    for i = 1:length(u)
        if u(i) >= lambda
            x(i) = u(i) - lambda;
        end
        if u(i) <= - lambda
            x(i) = u(i) + lambda;
        end
        if abs(u(i)) < lambda
            x(i) = 0;
        end
    end
    % update z
    z = x;

    
    % update theta
    theta = theta + z - X * alpha + y;
    
    cnt = cnt + 1;
    if sum(abs(z - X * alpha + y)) <= 0.1
        break
    end
end

display(sprintf('Finished in %d iterations', cnt));
alpha




% calculate the second half
w = ones(size(F, 2), 1);
z = F*w - Yp;
rho = 1;
theta = ones(size(z));

s = (F'*F) \F';

cnt = 0;
while true
    
    % update w
    w = s * (z + Yp + theta);
    
    u = F*w - Yp - theta;
    lambda = 1 / rho;
    x = zeros( size(z) );
    for i = 1:length(u)
        if u(i) >= lambda
            x(i) = u(i) - lambda;
        end
        if u(i) <= - lambda
            x(i) = u(i) + lambda;
        end
        if abs(u(i)) < lambda
            x(i) = 0;
        end
    end
    % update z
    z = x;
    
    % update theta
    theta = theta + z - F*w + Yp;
    
    cnt = cnt + 1;
    if sum(abs(z- F*w + Yp)) <= 0.01
        break
    end
end

display(sprintf('Finished in %d iterations', cnt));
w
