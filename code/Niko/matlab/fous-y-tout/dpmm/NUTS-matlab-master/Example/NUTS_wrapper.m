function [samples, logp_samples] = NUTS_wrapper(f, theta0, n_warmup, n_mcmc, delta)
% A wrapper function for the step-size adjustment and sampling stages of
% NUTS. 

if nargin < 5
    delta = .8; 
end

% The number of updates provided to the user during the sampling phase.
n_updates = 10;
n_itr_per_update = ceil(n_mcmc / n_updates);

% Adapt the step-size using dual-averaging algorithm.
n_updates_warmup = ceil(n_warmup / n_itr_per_update);
[theta, epsilon] = dualAveraging(f, theta0, delta, n_warmup, n_updates_warmup);

% Keep the post warm-up samples.
samples = zeros(length(theta0), n_mcmc);
logp_samples = zeros(n_mcmc, 1);

[samples(:,1), ~, nfevals_total, logp_samples(1), grad] = NUTS(f, epsilon, theta);
for i = 2:n_mcmc
    [samples(:,i), ~, nfevals, logp_samples(i), grad] = NUTS(f, epsilon, samples(:,i-1), logp_samples(i-1), grad);
    nfevals_total = nfevals_total + nfevals;
    if mod(i, n_itr_per_update) == 0
        fprintf('%d iterations have been completed.\n', i);
    end
end
fprintf('Each iteration of NUTS required %.1f gradient evaluations on average.\n', ...
    nfevals_total / n_mcmc);

end