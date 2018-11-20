function [gamma,lognormal, inverse_gamma] = gamma_vs_lognormal(mean,variance)


    %lognormal
    v = log1p(variance/mean^2);
    mu = log(mean) - v/2;
    
    lognormal = @(x) exp(-(log(x)-mu).^2/(2*v)) ./ (x*sqrt(v*2*pi));
    
    %gamma
    beta = mean/variance;
    alpha = beta*mean;
    gamma = @(x) beta^alpha * x.^(alpha-1) .* exp(-beta*x - gammaln(alpha));

    inverse_gamma = @(x) beta^alpha * x.^(-alpha-1) .* exp(-beta./x - gammaln(alpha));


end