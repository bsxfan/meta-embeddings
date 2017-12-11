function [E,back] = SGME_extract(w,zdim,R,nu)

    if nargin==0
        test_this();
        return;
    end
    
    
    if ~exist('nu','var') || isempty(nu)
        [E,back] = SGME_extract_nu_variable(w,zdim,R);
        return;
    end
    

    [rdim,~] = size(R);
    delta = rdim-zdim; 
    szP = rdim*zdim;
    
    P = reshape(w(1:szP),zdim,rdim);
    H = reshape(w(szP+1:end),delta,rdim);
    
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
        
        dA = dE(1:end-1,:);
        db = dE(end,:);
        
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
        
        dw = [dP(:);dH(:)];
        
    end
    
    

end


function test_this()

    zdim = 2;
    rdim = 4;
    n = 5;
    P = randn(zdim,rdim);
    H = randn(rdim-zdim,rdim);
    w = [P(:);H(:)];
    nu = pi;
    R = randn(rdim,n);
    
    f = @(w) SGME_extract(w,zdim,R,nu);
    
    testBackprop(f,w);

end
