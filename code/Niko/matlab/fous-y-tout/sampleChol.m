function Z = sampleChol(a,B,n,X)

    dim = length(a);
    
    if ~exist('X','var') || isempty(X)
        X = randn(dim,n);
    end
    
    
    
    
    R = chol(B);
    Z = R \ bsxfun(@plus,R.'\a,X);




end