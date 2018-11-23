function [logp, grad] = gradLogitPosterior(beta, y, X, prior)
% function [grad, logp] = gradLogitPosterior(beta, y, X, prior)
%
% Computes the log posterior probability of the regression coefficient
% 'beta' and its gradient for a Bayesian logistic regression model.
%
% Args:
% beta - column vector of length d+1 
% y - boolean vector (outcome variable)
% X - n by (d+1) matrix with an intercept at column index 1 (design matrix)
% prior - function handle to return log probability of the prior on 'beta'
%     (up to an additive constant) and its gradient

if nargin < 4
    % Default flat prior.
    prior = @(beta) deal(zeros(length(beta), 1), 0);
end

[grad_prior, logp_prior] = prior(beta);

pred_prob = 1 ./ (1 + exp(- X * beta));
loglik = zeros(length(y),1);
loglik(y) = log(pred_prob(y));
loglik(~y) = log(1 - pred_prob(~y));
logp = sum(loglik) - logp_prior;
grad = X' * (y - pred_prob) - grad_prior;

end