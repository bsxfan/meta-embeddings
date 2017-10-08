function [F,Z,labels,X,lambda] = rand_ivector(M,nu,W,alpha,K,T,minDur,maxDur,N)
% Inputs:
%   M: dim-by-L i-vector means for L languages
%   nu: degrees of freedom for i-vector t-distribution
%   W: within language precision
%   alpha: concentration parameter for Dirichlet for respnsibilities
%   K: UBM size
%   T: S-by-dim, factor loading matrix
%   minDur,maxDur: endpoints for uniform random duration sampling
%   N: number of i-vectors to generate

    if isempty(alpha)
        alpha = 1/(2*K);  % smaller alpha will give more concentrated (low entropy) distributions
    end

    [dim,L] = size(M);    % L: number of languages
    [S,dim2] = size(T);   % S: supervector dimension
    assert(dim==dim2); 
    
    fdim = S/K;
    
    R = chol(W);
    labels = sparse(randi(L,1,N),1:N,1,L,N);
    dur = randi([minDur,maxDur],1,N);
    
    lambda = randgamma(nu/2,nu/2,N);
    X = R\bsxfun(@rdivide,randn(dim,N),sqrt(lambda)) + M*labels;
    
    F = zeros(S,N);
    Z = zeros(K,N);
    for i=1:N
        resp = randDirichlet(alpha,K,dur(i));
        phi = randn(fdim,dur(i));
        Z(:,i) = sum(resp,2);
        Fi = bsxfun(@times,reshape(T*X(:,i),fdim,K),Z(:,i));
        F(:,i) = reshape(Fi + phi*resp.',S,1);
    end


end