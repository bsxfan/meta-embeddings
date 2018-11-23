function [ess, auto_cor] = ESS(x, mu, sigma_sq)
% function [ess, auto_cor] = ESS(x, mu, sigma_sq)
%
% Returns an estimate of effective sample sizes of a Markov chain 'x(:,i)'
% for each i. The estimates are based on the monotone positive sequence estimator
% of "Practical Markov Chain Monte Carlo" by Geyer (1992). The estimator is
% ONLY VALID for reversible Markov chains. The inputs 'mu' and 'sigma_sq' are optional.
%
% Examples:
% ess_for_mean = ESS(samples);
% ess_for_second_moment = ESS(samples.^2)
%
% Args:
% mu, sigma_sq - column vectors for the mean E(x) and variance Var(x) if the
%     analytical (or accurately estimated) value is available. If provided,
%     it can stabilize the estimate of auto-correlation of 'x(i,:)' and
%     hence its ESS. This is intended for research uses when one wants to
%     accurately quantify the asymptotic efficience of a MCMC algorithm.
%
% Returns:
% ess - column vector of length size(x, 1)
% auto_cor - cell array of length size(x, 1): the i-th cell contains the
%     auto-correlation estimate of 'x(i,:)' up to the lag at which the
%     auto-correlation can be considered insignificant by the monotonicity
%     criterion.

if nargin < 2
    mu = mean(x,2);
    sigma_sq = var(x,[],2);
end

d = size(x, 1);
ess = zeros([d, 1]);
auto_cor = cell(d, 1);
% Difficult to avoid the loop for this estimator of ESS.
for i = 1:d
    [ess(i), auto_cor{i}] = ESS_1d(x(i,:), mu(i), sigma_sq(i));
end

end

function [ess, auto_cor] = ESS_1d(x, mu, sigma_sq)

[~, n] = size(x);
auto_cor = [];

lag = 0;
even_auto_cor = computeAutoCorr(x, lag, mu, sigma_sq);
auto_cor(end + 1) = even_auto_cor;
auto_cor_sum = - even_auto_cor;

lag = lag + 1;
odd_auto_cor = computeAutoCorr(x, lag, mu, sigma_sq);
auto_cor(end + 1) = odd_auto_cor;
running_min = even_auto_cor + odd_auto_cor;

while (even_auto_cor + odd_auto_cor > 0) && (lag + 2 < n)

    running_min = min(running_min, (even_auto_cor + odd_auto_cor));
    auto_cor_sum = auto_cor_sum + 2 * running_min;

    lag = lag + 1;
    even_auto_cor = computeAutoCorr(x, lag, mu, sigma_sq);
    auto_cor(:, end + 1) = even_auto_cor;

    lag = lag + 1;
    odd_auto_cor = computeAutoCorr(x, lag, mu, sigma_sq);
    auto_cor(end + 1) = odd_auto_cor;
end

ess = n ./ auto_cor_sum;
if auto_cor_sum < 0 % Rare, but can happen when 'x' shows strong negative correlations.
    ess = Inf;
end

end

function [auto_corr] = computeAutoCorr(x, k, mu, sigma_sq)
% Returns an estimate of the lag 'k' auto-correlation of a time series
% ''. The estimator is biased towards zero due to the factor (n - k) / n.
% See Geyer (1992) Section 3.1 and the reference therein for justification.

n = length(x);
t = 1:(n - k);
auto_corr = (x(t) - mu) .* (x(t + k) - mu);
auto_corr = mean(auto_corr) / sigma_sq * ((n - k) / n);

end
