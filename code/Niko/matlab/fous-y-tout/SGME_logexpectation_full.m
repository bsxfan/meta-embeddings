function [y,back] = SGME_logexpectation_full(A,b,B)
% log expected values (w.r.t. standard normal) of diagonalized SGMEs
% Inputs:
%    A: dim-by-n, natural parameters (precision *mean) for n SGMEs    
%    b: 1-by-n, precision scale factors for these SGMEs
%    B: dim-by-dim, common precision (full) matrix factor 
%
% Note:
%    A(:,j) , b(j)*B forms the meta-embedding for case j
%
% Outputs:
%   y: 1-by-n, log expectations
%   back: backpropagation handle, [dA,db,dB] = back(dy)


    if nargin==0
        test_this();
        return;
    end

    [V,L] = eig(B);  %V*L*V' = B
    L = diag(L);
    bL = bsxfun(@times,b,L);
    logdets = sum(log1p(bL),1);
    bL1 = 1 + bL;
    S = V*bsxfun(@ldivide,bL1,V.'*A);
    Q = sum(A.*S,1);
    y = (Q-logdets)/2;
    


    back = @back_this;

    
    
    function [dA,db,dB] = back_this(dy)

        hdy = dy/2;
        dA = bsxfun(@times,hdy,S);
        dS = bsxfun(@times,hdy,A);
        dlogdets = -hdy;
        
        dA2 = V*bsxfun(@ldivide,bL1,V.'*dS);
        dA = dA + dA2;

        dBlogdet = V*bsxfun(@times,sum(bsxfun(@rdivide,b.*dlogdets,bL1),2),V.');
        dBsolve = bsxfun(@times,-b,dA2) * S.';
        dB = dBlogdet + (dBsolve+dBsolve.')/2;
        
        db = L.'*bsxfun(@ldivide,bL1,dlogdets) - sum(S.*(B*dA2),1);
        
        
        
        
    end


end

function test_this()

    m = 3;
    n = 5;
    A = randn(m,n);
    b = rand(1,n);
    B = randn(m,m+1); B = B*B.';

    fprintf('test function values:\n');
    err = max(abs(SGME_logexpectation_full(A,b,B)-SGME_logexpectation_slow(A,b,B))),
    
    fprintf('test derivatives:\n');
    [y,back] = SGME_logexpectation_full(A,b,B);
    dy = randn(size(y));
    [dAf,dbf,dBf] = back(dy);

    [~,back] = SGME_logexpectation_slow(A,b,B);
    [dAs,dbs,dBs] = back(dy);
    
    err_dA = max(abs(dAs(:)-dAf(:))),
    err_db = max(abs(dbs(:)-dbf(:))),
    err_dB = max(abs(dBs(:)-dBf(:))),
    
    
    %neither complex, nor real step differentiation seem to work through eig()
    %testBackprop(@SGME_logexpectation_full,{A,b,B},{1,0,1});

end





