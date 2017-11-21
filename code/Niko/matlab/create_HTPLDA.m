function HTPLDA = create_HTPLDA(F,nu,W)

    [rdim,zdim] = size(F);
    assert(rdim>zdim);
    nu_prime = nu + rdim - zdim;
    
    if ~exist('W','var') || isempty(W)
        W = speye(rdim);
    end
    
    E = F'*W*F;
    G = W - W*F*(E\F.')*W;

    [V,D] = eig(E);  % E = VDV'
    d = diag(D);     % eigenvalues
    VFW = V'*F*W;
    
    HTPLDA.extractGMEs = @extractGMEs;
    HTPLDA.log_expectations = @log_expectations;
    
    function [A,beta] = extractGMEs(R)
        q = sum(R.*(G*R),1);
        beta = nu_prime./(nu+q);
        A = bsxfun(@times,beta,VFW*R);
    end
    
    function y = log_expectations(A,beta)
        logdets = sum(log1p(bsxfun(@times,beta,d)),1);
        Q = sum((A.^2)./(1+bsxfun(@times,beta,d)),1);
        y = (Q-logdets)/2;
    end


    function Y = logLR(Left,Right)
    end
    
    

end