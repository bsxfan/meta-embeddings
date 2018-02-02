function [E,back] = SGME_extract_nu_variable(w,zdim,R)

    if nargin==0
        test_this();
        return;
    end

    [rdim,~] = size(R);
    delta = rdim-zdim; 
    szP = rdim*zdim;
    
    w1 = w(1);
    %nu = exp(w1);
    nu = w1.^2;
    P = reshape(w(2:szP+1),zdim,rdim);
    H = reshape(w(szP+2:end),delta,rdim);
    
    nuprime = nu + delta;

    HR = H*R;
    q = sum(HR.^2,1);
    den = nu + q;
    b = nuprime./den;

    M = P*R;
    A = bsxfun(@times,b,M);
    
    
    E = [A;b];
    
    back = @back_this;
    
    
    function dw = back_this(dE)
        
        %E = [A;b];
        dA = dE(1:end-1,:);
        db = dE(end,:);
        
        %A = bsxfun(@times,b,M);
        db = db + sum(dA.*M,1);
        dM = bsxfun(@times,b,dA);
        
        %M = P*R;
        dP = dM*R.';
        
        %b = nuprime./den;
        dnuprime = sum(db./den);
        dden = -db.*b./den;
        
        %den = nu + q;
        dnu = sum(dden);
        dq = dden;
        
        %q = sum(HR.^2,1);
        dHR = bsxfun(@times,2*dq,HR);
        
        %HR = H*R;
        dH = dHR*R.';
        
        %nuprime = nu + delta;
        dnu = dnu + dnuprime;
        
        %nu = exp(w1);
        %dw1 = nu*dnu;
        %nu = w1.^2;
        dw1 = 2*w1*dnu;
        
        dw = [dw1;dP(:);dH(:)];
        
    end
    
    

end


function test_this()

    zdim = 2;
    rdim = 4;
    n = 5;
    P = randn(zdim,rdim);
    H = randn(rdim-zdim,rdim);
    nu = rand;
    w = [nu;P(:);H(:)];
    R = randn(rdim,n);
    
    f = @(w) SGME_extract_nu_variable(w,zdim,R);
    
    testBackprop(f,w);

end
