function [y,back] = SGME_logexpectation(A,b,d)
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

    

    bd = bsxfun(@times,b,d);
    logdets = sum(log1p(bd),1);
    den = 1 + bd;
    Aden = A./den;
    Q = sum(A.*Aden,1);    %Q = sum((A.^2)./den,1);
    y = (Q-logdets)/2;

    back = @back_this;


    function [dA,db,dd] = back_this(dy)
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
        
    end


end

function test_this0()

    m = 3;
    n = 5;
    A = randn(m,n);
    b = rand(1,n);
    d = rand(m,1);
    
    
    testBackprop(@SGME_logexpectation,{A,b,d},{1,1,1});

end





function test_this()

    %em = 4;
    n = 7;
    dim = 2;
    
    %prior = create_PYCRP([],0,em,n);
    %poi = prior.sample(n);
    %m = max(poi);
    %blocks = sparse(poi,1:n,true,m+1,n);  
    %num = find(blocks(:));    
    
    %logPrior = prior.GibbsMatrix(poi);  

    d = rand(dim,1);
    A = randn(dim,n);
    b = rand(1,n);
    
    
    f = @(A,b,d) SGME_logexpectation(A,b,d);
    testBackprop(f,{A,b,d},{1,1,1});

    
    
    
    
    
end
