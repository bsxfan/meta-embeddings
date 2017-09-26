function [Y,back] = dsolve(RHS,A)
% SOLVE: Y= A\RHS, with backpropagation into both arguments 
%
% This is mostly for debugging purposes. It can be done more efficiently 
% by caching a matrix factorization to re-use for derivative (and also for 
% the determinant if needed).  


    if nargin==0
        test_this();
        return;
    end

    Y = A\RHS;
    
    back = @back_this;
    
    function [dRHS,dA] = back_this(dY)
        dRHS = A.'\dY;   % A\dY = dsolve(dY,A) can be re-used for symmetric A
        if nargout>=2
            dA = -dRHS*Y.';
        end
    end

end


% function [Y,back] = IbetaB(beta,B)
%     dim = size(B,1);
%     Y = speye(dim)+beta*B;
%     back = @(dY) trace(dY*B.');
% end


function test_this()

    m = 5;
    n = 2;
    A = randn(m,m);
    RHS = randn(m,n);
    
    testBackprop(@dsolve,{RHS,A});
    testBackprop_rs(@dsolve,{RHS,A},1e-4);
    
%     beta = rand/rand;
%     testBackprop(@(beta) IbetaB(beta,A),{beta});
    
    

end