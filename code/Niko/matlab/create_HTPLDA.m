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
    
    HTPLDA.extractSGMEs = @extractSGMEs;
    HTPLDA.log_expectations = @log_expectations;
    HTPLDA.logLR = @logLR;
    
    function [A,b] = extractSGMEs(R)
        q = sum(R.*(G*R),1);
        b = nu_prime./(nu+q);
        A = bsxfun(@times,b,VFW*R);
        sgme.b = b;
        sgme.A = A;
    end
    
    function y = log_expectations(A,b)
        bd = bsxfun(@times,b,d);
        logdets = sum(log1p(bd),1);
        Q = sum((A.^2)./(1+bd),1);
        y = (Q-logdets)/2;
    end


    function Y = logLR(left,right)
        B = bsxfun(@plus,left.b.',right.b);
        [m,n] = size(B);
        Y = zeros(m,n);
        for i=1:m
            AA = bsxfun(@plus,left.A(:,i),right.A);
            Y(i,:) = log_expectations(AA,B(i,:));
        end
    end
    
    

end