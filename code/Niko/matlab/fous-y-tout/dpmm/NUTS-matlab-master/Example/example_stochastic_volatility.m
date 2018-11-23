%% Load the closing values of S&P 500 index for 3000 days ending on Dec 31st, 2015.

% The observation on Jan 2nd, 2008 is removed since it otherwise leads to a
% challenging multi-modal posterior. This is because the price on Jan 2nd, 
% 2008 happens to be exactly the same as the previously observed value and
% consequently the posterior of the corresponding volatility parameter 
% 'sigma' has a local mode around zero. HMC and NUTS in general have 
% troubles in the presense of severe multi-modaility.
load('SP500.mat', 'y')
outlier_index = 1008; 
y(outlier_index) = [];
logy = log(y);
logy_diff = logy(2:end) - logy(1:(end-1));

f = @(logsigma) gradSvPosterior(logsigma, logy_diff);
logsigma0 = zeros(length(logy_diff), 1);

%% Sample from the posterior with NUTS and plot ESS's
n_warmup = 300;
n_mcmc_samples = 3000;
[samples, logp_samples] = NUTS_wrapper(f, logsigma0, n_warmup, n_mcmc_samples);

% Compute effective sample sizes (ESS) for each parameter. 
ess_mean = ESS(samples);
ess_sec_moment = ESS(samples.^2); 

set(0,'defaultAxesFontSize', 18) 
marker_size = 6;
plot(ess_mean, 'o', 'MarkerSize', marker_size)
hold on
plot(ess_sec_moment, 'x', 'MarkerSize', marker_size)
refline(0, size(samples, 2))
legend('for mean', 'for 2nd moment')
xlabel('Parameters')
ylabel('ESS')
title('Effective sample sizes')
hold off