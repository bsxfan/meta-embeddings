%% Define a gradient function for a 'd'-dimensional Gaussian.
d = 100;
deg_freedom = ceil(1.5 * d);

% Deraw a random covariance matrix.
Phi = wishrnd(eye(d), deg_freedom);

gaussian_logp = @(x) - x' * Phi * x / 2;
gaussian_grad = @(x) -  Phi * x;
f = @(x) deal(gaussian_logp(x), gaussian_grad(x));
theta0 = randn(d, 1);

%% Adjust stepsize via dual averaging.
%  The stepsize of the leap-frog method (for approximating solutions of
%  Hamiltonian dynamics) is a tuning parameter of NUTS. With dual-averaging 
%  / stochastic optimization algorithm, the task of choosing the step-size
%  is reduced to the choice of a parameter 'delta', which roughly 
%  can be thought of as the acceptance rate of a proposal. (Only 'roughly',
%  because NUTS employs a type of delayed-rejection scheme and there is no 
%  single acceptance and rejection step.)
%
%  A value of 'delta' in the range 0.6 ~ 0.9 are reasonable, and a higher 
%  "acceptance rate" comes with a higher computatinal cost for one 
%  iteration of NUTS. A value toward the higher end of the range is a safer 
%  choice for models with complex interactions among the parameters. The 
%  default value in Stan is 0.8 and seems to be a good enough choice in 
%  most situations.
%
%  The number of iterations needed to find a good step-size varies
%  substantially from one model to another. For models with simple
%  posterior distributions (e.g. logistic regression), 50 iterations may
%  well be more than enough. But more complex posterior distributions may 
%  require a few hundreds or more iterations. See the Stan manual for
%  useful tricks (pre-prossessing of parameters) to accelerate the mixing 
%  of NUTS in general (and hence in particular reduce the number of
%  iterations needed for tuning.

seed = 1;
rng(seed)
n_warmup = 100;
delta = .8; 

% Run NUTS and adapt its step-size using dual-averaging algorithm.
[theta, epsilon, epsilon_seq, epsilonbar_seq] = dualAveraging(f, theta0, delta, n_warmup);

set(0,'defaultAxesFontSize', 18) 
plot(epsilon_seq)
hold on
plot(epsilonbar_seq)
title('Stepsize adaptation via dual-averaging')
xlabel('Iteration')
ylabel('Stepsize')
legend('Attempted value at each iteration', 'Running (weighted) average')
hold off

%% Run NUTS with a fixed stepsize to generate posterior samples.
n_mcmc = 2500;
n_updates = 10;
n_itr_per_update = ceil(n_mcmc / n_updates);
samples = zeros(length(theta), n_mcmc);
logp_samples = zeros(n_mcmc, 1);

[samples(:,1), ~, nfevals_total, logp_samples(1)] = NUTS(f, epsilon, theta);
for i = 2:n_mcmc
    [samples(:,i), ~, nfevals, logp_samples(i)] = NUTS(f, epsilon, samples(:,i-1));
    nfevals_total = nfevals_total + nfevals;
    if mod(i, n_itr_per_update) == 0
        fprintf('%d iterations have been completed.\n', i);
    end
end
fprintf('Each iteration of NUTS required %.1f gradient evaluations on average.\n', ...
    nfevals_total / n_mcmc);

% Basic convergence diagnostic.
set(0,'defaultAxesFontSize', 18) 
plot(logp_samples)
xlabel('Iterations')
ylabel('Log probability')
title('Traceplot of $\log(\pi(\theta))$', 'Interpreter', 'LaTex')

%% Compute effective sample sizes (ESS) for each parameter. 
% The estimator used here is one of the most reliable and provide estimates
% that are generally in the right ballpark (as long as the length of a
% chain is much longer than the lag it takes for the auto-correlation to
% become negligible).

set(0,'defaultAxesFontSize', 18) 
marker_size = 6;
ess_mean = ESS(samples);
ess_sec_moment = ESS(samples.^2); 
plot(ess_mean, 'o', 'MarkerSize', marker_size)
hold on
plot(ess_sec_moment, 'x', 'MarkerSize', marker_size)
refline(0, size(samples, 2))
legend('for mean', 'for 2nd moment')
xlabel('Parameters')
ylabel('ESS')
title('Effective sample sizes')
hold off
