function [A,b,back] = SGME_extract(P,H,nu,R)

    if nargin==0
        test_this();
        return;
    end

    [zdim,rdim] = size(P);
    nuprime = nu + rdim - zdim;

    HR = H*R;
    q = sum(HR.^2,1);
    den = nu + q;
    b = nuprime./den;

    M = P*R;
    A = bsxfun(@times,b,M);
    
    
    back = @back_this;
    
    
    function [dP,dH] = back_this(dA,db)
        
        %A = bsxfun(@times,b,M);
        db = db + sum(dA.*M,1);
        dM = bsxfun(@times,b,dA);
        
        %M = P*R;
        dP = dM*R.';
        
        %b = nuprime./den;
        dden = -db.*b./den;
        
        %den = nu + q;
        dq = dden;
        
        %q = sum(HR.^2,1);
        dHR = bsxfun(@times,2*dq,HR);
        
        %HR = H*R;
        dH = dHR*R.';
        
        
        
    end
    
    

end


function test_this()

    zdim = 2;
    rdim = 4;
    n = 5;
    P = randn(zdim,rdim);
    H = randn(rdim-zdim,rdim);
    
    nu = pi;
    R = randn(rdim,n);
    
    f = @(P,H) SGME_extract(P,H,nu,R);
    
    testBackprop_multi(f,2,{P,H});

end
