function [HTPLDA,V] = create_HTPLDA(F,nu,W)

    if nargin==0
        test_this();
        return;
    end

    [rdim,zdim] = size(F);
    assert(rdim>zdim);
    nu_prime = nu + rdim - zdim;
    
    if ~exist('W','var') || isempty(W)
        W = speye(rdim);
    end
    
    E = F.'*W*F;
    G = W - W*F*(E\F.')*W;

    [V,D] = eig(E);  % E = VDV'
    d = diag(D);     % eigenvalues
    VFW = V.'*F.'*W;
    
    HTPLDA.extractSGMEs = @extractSGMEs;
    HTPLDA.SGME2GME = @SGME2GME;
    HTPLDA.log_expectations = @log_expectations;
    HTPLDA.logLR = @logLR;
    
    ii = reshape(logical(eye(zdim)),[],1);
    
    
    
    function [A,b] = extractSGMEs(R)
        q = sum(R.*(G*R),1);
        b = nu_prime./(nu+q);
        A = bsxfun(@times,b,VFW*R);
    end
    
    function [A,B,V1] = SGME2GME(A,b)
        B = zeros(zdim*zdim,length(b));
        B(ii,:) = bsxfun(@times,b,d);
        V1 = V;
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

function test_this()

    zdim = 2;
    xdim = 20;      %required: xdim > zdim
    nu = 2;         %required: nu >= 1, integer, DF
    fscal = 2;      %increase fscal to move speakers apart
    
    F = randn(xdim,zdim)*fscal;

    
    HTPLDA = create_HTPLDA(F,nu);
    
    z1 = randn(zdim,1);  %speaker 1 
    z2 = randn(zdim,1);  %speaker 2
    Z = [z1,z2,z2];      %1 of speaker 1 and 2 of speaker 2
    
    
    [R,lambda] = sample_speaker(Z,F,nu/2,[],true);

    [As,b] = HTPLDA.extractSGMEs(R);
    [A,B] = HTPLDA.SGME2GME(As,b);
    

    [lambda;b]
    
    [plain_GME_log_expectations(A,B);HTPLDA.log_expectations(As,b)]
    
    
    
end

