function [y,back] = posteriorNorm_fast(A,B,b)
% Computes, for every i:  log N( 0 | Pi\A(:,i), inv(Pi) ), where
%
%   precisions are Pi = I + b(i)*B
%
% This is the fast version, which simultaneously diagonalizes all the Pi,
% using eigenanalysis of B.
%
% Inputs:
%    A: dim-by-n, natural parameters (precision *mean) for n Gaussians    
%    B: dim-by-dim, common precision matrix factor (full, positive semi-definite)
%    b: 1-by-n, precision scale factors 
%
% Outputs:
%   y: 1-by-n, log densities, evaluated at zero
%   back: backpropagation handle, [dA,dB,db] = back(dy)


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
    y = (logdets - Q)/2;
    


    back = @back_this;

    
    
    function [dA,dB,db] = back_this(dy)

        hdy = dy/2;
        dA = bsxfun(@times,-hdy,S);
        dS = bsxfun(@times,-hdy,A);
        dlogdets = hdy;
        
        dA2 = V*bsxfun(@ldivide,bL1,V.'*dS);
        dA = dA + dA2;

        dBlogdet = V*bsxfun(@times,sum(bsxfun(@rdivide,b.*dlogdets,bL1),2),V.');
        dBsolve = bsxfun(@times,-b,dA2) * S.';
        dB = dBlogdet + (dBsolve+dBsolve.')/2;
        
        if nargout>=3
            db = L.'*bsxfun(@ldivide,bL1,dlogdets) - sum(S.*(B*dA2),1);
        end
        
        
        
        
    end


end

function test_this()

    m = 3;
    n = 5;
    A = randn(m,n);
    b = rand(1,n);
    B = randn(m,m+1); B = B*B.';

    fprintf('test function values:\n');
    err = max(abs(posteriorNorm_fast(A,B,b) - posteriorNorm_slow(A,B,b))),
    
    fprintf('test derivatives:\n');
    [y,back] = posteriorNorm_fast(A,B,b);
    dy = randn(size(y));
    [dAf,dBf,dbf] = back(dy);

    [~,back] = posteriorNorm_slow(A,B,b);
    [dAs,dBs,dbs] = back(dy);
    
    err_dA = max(abs(dAs(:)-dAf(:))),
    err_db = max(abs(dbs(:)-dbf(:))),
    err_dB = max(abs(dBs(:)-dBf(:))),
    
    
    %neither complex, nor real step differentiation seem to work through eig()

end

