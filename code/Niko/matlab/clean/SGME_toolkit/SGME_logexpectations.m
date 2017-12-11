function [y,back] = SGME_logexpectations(E,w)
% log expected values (w.r.t. standard normal) of diagonalized SGMEs
% Inputs:
%    A: dim-by-n, natural parameters (precision *mean) for n SGMEs    
%    b: 1-by-n, precision scale factors for these SGMEs
%    d: dim-by-1, common diagonal precision 
%
% Note:
%    bsxfun(@times,b,d) is dim-by-n precision diagonals for the n SGMEs 
%
% Outputs:
%   y: 1-by-n, log expectations
%   back: backpropagation handle, [dA,db,dd] = back(dy)


    if nargin==0
        test_this();
        return;
    end

    
    A = E(1:end-1,:);
    b = E(end,:);
    
    d = w.^2;
    
    
    bd = bsxfun(@times,b,d);
    logdets = sum(log1p(bd),1);
    den = 1 + bd;
    Aden = A./den;
    Q = sum(A.*Aden,1);    %Q = sum((A.^2)./den,1);
    y = (Q-logdets)/2;

    back = @back_this;


    function [dE,dw] = back_this(dy)
        dQ = dy/2;
        %dlogdets = - dQ;

        dAden = bsxfun(@times,dQ,A);           
        dA = bsxfun(@times,dQ,Aden);           

        dA2 = dAden./den;
        dA = dA + dA2;          
        dden = -Aden.*dA2;

        dbd = dden - bsxfun(@rdivide,dQ,den);    %dlogdets = -dQ

        db = d.' * dbd;
        dd = dbd * b.';
        
        dE = [dA;db];
        
        dw = 2*w.*dd;
        
    end


end




function test_this()

    n = 7;
    dim = 2;
    

    w = randn(dim,1);
    A = randn(dim,n);
    b = rand(1,n);
    
    E = [A;b];
    
    f = @(E,w) SGME_logexpectations(E,w);
    testBackprop(f,{E,w});

    
    
    
    
    
end
