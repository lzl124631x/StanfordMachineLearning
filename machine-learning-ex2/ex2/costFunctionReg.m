function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
T=theta;
T(1) = 0;
H = sigmoid(X * theta);
J = -1/m * sum((y.*log(H)) .+ (1-y).*log(1-H)) + lambda / 2 / m * sum(T.^2);
grad = 1/m * (X' * (H - y)) + lambda / m * T;

end
