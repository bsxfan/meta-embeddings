function G = randg(alpha,m,n)
% Generates an m-by-n matrix of random gamma variates, having scale = 1
% and shape alpha. 
% Inputs: 
%   alpha: scalar, vector or matrix
%   m,n: [optional] size of output matrix. If not given, the size is the
%   same as that of alpha. If given, then alpha should be m-by-n, or an 
%   m-vector, or an n-row.
% Output:
%    G: m-by-n matrix of gamma variates
%
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

   
    if nargin==0
        test_this();
        return;
    end

    if exist('m','var')
        assert(exist('n','var')>0,'illegal argument combination');
        [mm,nn] = size(alpha);
        if isscalar(alpha)
            alpha = alpha*ones(m,n);
        elseif nn==1 && mm==m && n>1
            alpha = repmat(alpha,1,n);
        elseif mm==1 && nn==n && m>1
            alpha = repmat(alpha,m,1);
        else
            assert(m==mm && n==nn,'illegal argument combination');
        end
        
    else
        [m,n] = size(alpha);
    end
    N = m*n;
    alpha = reshape(alpha,1,N);


    G = zeros(1,N);
    req = 1:N;

    small =  alpha < 1;
    if any(small)
        ns = sum(small);
        sa = alpha(small);
        G(small) = randg(1+sa,1,ns).*rand(1,ns).^(1./sa);
        req(small)=[];
    end
    nreq = length(req);
    
    
    d = alpha(req) - 1/3; 
    c = 1./sqrt(9*d);
    
    while nreq>0
        
        
        x = randn(1,nreq);
        v = (1+c.*x).^3;
        u = rand(1,nreq);
        ok = ( v>0 ) & ( log(u) < x.^2/2 + d.*(1 - v + log(v)) );
        G(req(ok)) = d(ok).*v(ok);
        
        d(ok) = [];
        c(ok) = [];
        req(ok) = [];
        nreq = length(req);
        
        % This version works nicely, but can be made faster (fewer logs), 
        % at the cost slightly more complex code---see the paper.
        
    end
    
    G = reshape(G,m,n);
    
    
    
    
end

function test_this()

    alpha = [0.1,1,10];
    G = randg(repmat(alpha,10000,1));
    mu = mean(G,1);
    v = mean(bsxfun(@minus,G,alpha).^2,1);
    [alpha;mu;v]

    alpha = pi;
    G = randg(alpha,10000,1);
    mu = mean(G,1);
    v = mean(bsxfun(@minus,G,alpha).^2,1);
    [alpha;mu;v]

end

