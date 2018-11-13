function G = randg(alpha,m,n)
% Generates an m-by-n matrix of random gamma variates, having scale = 1
% and shape alpha. 
% 
% Uses the method of:
% Marsaglia & Tsang, "A Simple Method for Generating Gamma Variables",
% 2000, ACM Transactions on Mathematical Software. 26(3). 
%
% See also:
% https://en.wikipedia.org/wiki/Gamma_distribution#Generating_gamma-
% distributed_random_variables
%
% The speed is roughly independent of alpha.

   


    if ~exist('m','var')
        m = 1;
    end

    if ~exist('n','var')
        n = 1;
    end

    if alpha < 1
        assert(alpha>0);
        G = randg(1+alpha,m,n).*rand(m,n).^(1/alpha);
        return;
    end
    
    d = alpha - 1/3; 
    c = 1/sqrt(9*d);
    N = m*n;
    G = zeros(1,N);
    req = 1:N;
    nreq = N;
    
    while nreq>0
        x = randn(1,nreq);
        v = (1+c*x).^3;
        u = rand(1,nreq);
        ok = ( v>0 ) & ( log(u) < x.^2/2 + d*(1- v + log(v)) );
        G(req(ok)) = d*v(ok);
        req = req(~ok);
        nreq = length(req);
    end
    
    G = reshape(G,m,n);
    
    
    
    
end