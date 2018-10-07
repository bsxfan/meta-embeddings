function X = sample_HTnoise(nu,dim,n,W)
% Sample n heavy-tailed dim-dimensional variables. (Only for integer nu.)
%
% Inputs:
%   nu: integer nu >=1, degrees of freedom of resulting t-distribution
%   n: number of samples
%   W: precision matrix for T-distribution
%
% Output:
%   X: dim-by-n samples

    cholW = chol(W);    
    if isinf(nu)
        precisions = ones(1,n);
    else
        precisions = mean(randn(nu,n).^2,1);
    end
    std = 1./sqrt(precisions);
    X = cholW\bsxfun(@times,std,randn(dim,n));
end
