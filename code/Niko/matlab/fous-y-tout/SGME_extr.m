function [A,b,d,reg,back] = SGME_extr(T,F,nu,R)

    if nargin==0
        test_this();
        return;
    end

    [zdim,rdim] = size(F);
    nuprime = nu + rdim - zdim;

    TR = T*R;
    
    
    A0 = F.'*TR;
    B0 = F.'*F;
    d = diag(B0);
    
    reg = (B0(:).'*B0(:) - d.'*d)/2;  %tr([B0-diag(d)]^2)
    
    DA = bsxfun(@ldivide,d,A0);
    q = sum(TR.^2,1) - sum(A0.*DA,1);

    den = nu + q;
    b = nuprime./den;

    A = bsxfun(@times,b,A0);
    
    
    back = @back_this;
    
    
    function [dT,dF] = back_this(dA,db,dd,dreg)
        
        %A = bsxfun(@times,b,A0);
        db = db + sum(dA.*A0,1);
        dA0 = bsxfun(@times,b,dA);
        
        %b = nuprime./den;
        dden = -db.*b./den;

        %den = nu + q;
        dq = dden;

        %q = sum(TR.^2,1) - sum(A0.*DA,1);
        dTR = bsxfun(@times,2*dq,TR);
        dA0 = dA0 - bsxfun(@times,dq,DA);
        dDA = bsxfun(@times,-dq,A0);
        
        
        %DA = bsxfun(@ldivide,d,A0);
        dA0 = dA0 + bsxfun(@ldivide,d,dDA);
        dd = dd - sum(dDA.*DA,2)./d;
        
        %reg = (B0(:).'*B0(:) - d.'*d)/2;
        dd = dd - dreg*d;
        dB0 = dreg*B0;
        

        %d = diag(B0);
        dB0 = dB0 + diag(dd);
        
        %B0 = F.'*F;
        dF = 2*F*dB0;  
        
        
        %A0 = F.'*TR;
        dF = dF + TR*dA0.';
        dTR = dTR + F*dA0;
        
        %TR = T*R;
        dT = dTR*R.';
        
        
        
        
        
    end
    
    

end


function test_this()

    zdim = 2;
    rdim = 4;
    n = 5;
    F = randn(rdim,zdim);
    T = randn(rdim,rdim);
    
    nu = pi;
    R = randn(rdim,n);
    
    f = @(T,F) SGME_extr(T,F,nu,R);
    
    testBackprop_multi(f,4,{T,F});

end
