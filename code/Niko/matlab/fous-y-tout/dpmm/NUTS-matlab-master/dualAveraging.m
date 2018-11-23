function [theta, epsilonbar, epsilon_seq, epsilonbar_seq] = dualAveraging(f, theta0, delta, n_warmup, n_updates)
% function [theta, epsilonbar, epsilon_seq, epsilonbar_seq] = dualAveraging(f, n_warmup, theta0, delta, n_updates)
%
% Adjusts the step-size of NUTS by the dual-averaging (stochastic 
% optimization) algorithm of Hoffman and Gelman (2014).
%
% Args:
% f - function handle: returns the log probability of the target and 
%     its gradient.
% theta0 - column vector: the initial state of the chain
% delta - double in the range [0, 1]: the target "acceptance rate" of NUTS.
% n_warmup - int: the number of NUTS iterations for the dual-averaging
%     algorithm. 
% n_updates - int: the number of updates to be printed till the completion
%     of the step-size adjustment.
%
% Returns:
% theta - last state of the chain after the step-size adjustment, which
%     can be used as the initial state for the sampling stage (no need for
%     additional 'burn-in' samples
% epsilonbar - step-size corresponding to the target "acceptance rate".
% epsilon_seq, epsilonbar_seq - column vectors: the whole history of the 
%     attempted and averaged step-size. It can be used to diagnose the 
%     convergence of the dual-averaging algorithm.

% Default argment values.
if nargin < 3
    delta = 0.8;
end

if nargin < 4
    n_warmup = 500;
end

if nargin < 5
    n_updates = 5; 
end

% Calculate the number of iterations per update.
n_itr_per_update = floor(n_warmup / n_updates);

[logp, grad] = f(theta0);

% Parameters for NUTS
max_tree_depth = 12;

% Choose a reasonable first epsilon by a simple heuristic.
[epsilon, nfevals_total] = find_reasonable_epsilon(theta0, grad, logp, f);

% Parameters for the dual averaging algorithm.
gamma = .05;
t0 = 10;
kappa = 0.75;
mu = log(10 * epsilon);

% Initialize dual averaging algorithm.
epsilonbar = 1;
epsilon_seq = zeros(n_warmup, 1);
epsilonbar_seq = zeros(n_warmup, 1);
epsilon_seq(1) = epsilon;
Hbar = 0;

theta = theta0;
for i = 1:n_warmup
    
    [theta, ave_alpha, nfevals, logp, grad] = NUTS(f, epsilon, theta, logp, grad, max_tree_depth);
    nfevals_total = nfevals_total + nfevals;
    eta = 1 / (i + t0);
    Hbar = (1 - eta) * Hbar + eta * (delta - ave_alpha);
    epsilon = exp(mu - sqrt(i) / gamma * Hbar);
    epsilon_seq(i) = epsilon;
    eta = i^-kappa;
    epsilonbar = exp((1 - eta) * log(epsilonbar) + eta * log(epsilon));
    epsilonbar_seq(i) = epsilonbar;
    
    % Update on the progress of simulation.
    if mod(i, n_itr_per_update) == 0
        disp(['The ', num2str(i), ' iterations are complete.'])
    end
    
end

fprintf('Each iteration of NUTS required %.1f gradient evaluations during the step-size adjustment.\n', ...
    nfevals_total / (n_warmup + 1));

end

function [epsilon, nfevals] = find_reasonable_epsilon(theta0, grad0, logp0, f)

epsilon = 1; 
r0 = randn(length(theta0), 1);
% Figure out what direction we should be moving epsilon.
[~, rprime, ~, logpprime] = leapfrog(theta0, r0, grad0, epsilon, f);
nfevals = 1;
acceptprob = exp(logpprime - logp0 - 0.5 * (rprime' * rprime - r0' * r0));
a = 2 * (acceptprob > 0.5) - 1;
% Keep moving epsilon in that direction until acceptprob crosses 0.5.
while (acceptprob^a > 2^(-a))
    epsilon = epsilon * 2^a;
    [~, rprime, ~, logpprime] = leapfrog(theta0, r0, grad0, epsilon, f);
    nfevals = nfevals + 1;
    acceptprob = exp(logpprime - logp0 - 0.5 * (rprime' * rprime - r0' * r0));
end

end

function [thetaprime, rprime, gradprime, logpprime] = leapfrog(theta, r, grad, epsilon, f)

rprime = r + 0.5 * epsilon * grad;
thetaprime = theta + epsilon * rprime;
[logpprime, gradprime] = f(thetaprime);
rprime = rprime + 0.5 * epsilon * gradprime;

end
