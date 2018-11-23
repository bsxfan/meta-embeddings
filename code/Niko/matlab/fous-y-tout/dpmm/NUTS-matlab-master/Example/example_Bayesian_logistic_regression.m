%% Load the data set and define a gradient function for the posterior of a 
%  Bayesian logistic regression model. 
load('german.mat', 'y', 'X')
[n, d] = size(X);

% Standardize the predictors. Putting the parameters (and hence the
% posterior variances) in the same scale helps the speed and mixing of
% NUTS.
X = (X - repmat(mean(X,1), n, 1)) ./ repmat(std(X,[],1), n, 1);
X = [ones(n,1), X];
f = @(beta) gradLogitPosterior(beta, y, X);
theta0 = zeros(d+1, 1);

%% Sample from the posterior with NUTS and plot ESS's
n_warmup = 100;
n_mcmc_samples = 1000;
[samples, logp_samples] = NUTS_wrapper(f, theta0, n_warmup, n_mcmc_samples);

% Compute effective sample sizes (ESS) for each parameter. 
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
%hold off