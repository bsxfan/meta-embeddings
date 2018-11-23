function [logp, grad] = gradSvPosterior(logsigma, logy_diff)
% Returns the log probability and its gradient of the stochastic
% volatility model below.
% 
% Model:
% log( y(t) / y(t-1) ) ~ Normal(mean = 0, sd = sigma(t))
% log( sigma(t) / sigma(t-1) ) ~ Cauchy(scale = kappa)
%     (This is equivalent to assuming the distribution 
%          log( sigma(t) / sigma(t-1) ) ~ Normal(0, kappa^2 * tau^-1)
%      and placing a prior tau ~ Gamma(1/2, 1/2). The hyper-parameter 
%      'kappa' controlls the prior scale of sigma's.)
% y(1) ~ Normal(sd = sigma(1))
%
% Prior:
% sigma(1) ~ Exp(mean = 1/lambda)
%
% Args:
% logsigma - column vector of length n representing the log of the 
%     volatility parameters 'sigma' where (n + 1) is the length of the 
%     observed time series 'y'.
% logy_diff - column vector of length n and is defined as 
%     logy_diff = log(y(2:(n+1))) - log(y(1:n)).

% Specify the hyper-parameters.
lambda = 10; 
kappa = 1 / 10;

n = length(logsigma);
logsigma_diff = logsigma(2:n) - logsigma(1:(n-1));
phi_sq = exp(- 2 * logsigma);

logp = - lambda * exp(logsigma(1)) + logsigma(1) - sum(logsigma(2:n)) ...
    - sum( log(kappa^2 + logsigma_diff.^2) ) ...
    - sum( logy_diff.^2 .* phi_sq ) / 2;

grad_s1 = - lambda * exp(logsigma(1)) + 1 ...
    + 2 * logsigma_diff(1) / (kappa^2 + logsigma_diff(1)^2) ...
    + logy_diff(1)^2 * phi_sq(1);
grad_sn = - 1 ...
    - 2 * logsigma_diff(n-1) / (kappa^2 + logsigma_diff(n-1)^2) ...
    + logy_diff(n)^2 * phi_sq(n);
grad_else = - 1 ...
    + 2 * logsigma_diff(2:(n-1)) ./ (kappa^2 + logsigma_diff(2:(n-1)).^2) ...
    - 2 * logsigma_diff(1:(n-2)) ./ (kappa^2 + logsigma_diff(1:(n-2)).^2) ...
    + logy_diff(2:(n-1)).^2 .* phi_sq(2:(n-1));
grad = [grad_s1; grad_else; grad_sn];

end