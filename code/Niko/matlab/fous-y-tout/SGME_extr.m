function [A,b,d,reg,back] = SGME_extr(T,F,H,nu,R)

    if nargin==0
        test_this();
        return;
    end

    
    [rdim,zdim] = size(F);
    nuprime = nu + rdim - zdim;

    TR = T*R;
    
    
    A0 = F.'*TR;
    B0 = F.'*F;
    d = diag(B0);
    
    
    %G = speye(rdim) - F*bsxfun(@ldivide,d,F.');
    %HH = H.'*H  This must be regularized to equal G
    
    
    %B0 must be diagonal
    delta1 = mdot(B0) - d.'*d;
    reg1 = 0.5*delta1^2;  %tr([B0-diag(d)]^2)
    
    %G=HH
    BF = bsxfun(@ldivide,d,F.');  % inv(B)*F'
    FFB = F.'*BF.';  % F'*F*inv(B)
    HF = H*F;
    BFH = BF*H.';    % inv(B)*F.'*H.'
    HH = H*H.';      
    trGG = rdim  -  2*mdot(F,BF.') + mdot(FFB);
    trGHH = mdot(H) - mdot(HF,BFH.');
    trHHHH = mdot(HH);
    reg2 = (trGG - 2*trGHH + trHHHH)/2;
    
    %H'H*F = 0
    HHHF = HH*HF;
    reg3 = mdot(HHHF,HF);
    
    %tr(H'H)=D-d
    delta4 = mdot(H) - rdim + zdim;
    reg4 = 0.5*delta4^2;
    
    
    reg = reg1 + reg2 + reg3 + reg4;
    
    HTR = H*TR;
    q = sum(HTR.^2,1);

    den = nu + q;
    b = nuprime./den;

    A = bsxfun(@times,b,A0);
    
    
    
    back = @back_this;
    
    
    function [dT,dF,dH] = back_this(dA,db,dd,dreg)
        
        
        %A = bsxfun(@times,b,A0)
        db = db + sum(dA.*A0,1);
        dA0 = bsxfun(@times,b,dA);
        
        %b = nuprime./den
        dden = -db.*b./den;

        %den = nu + q
        dq = dden;

        %q = sum(HTR.^2,1)
        dHTR = bsxfun(@times,2*dq,HTR);
        
        
        %HTR = H*TR
        dH = dHTR*TR.';
        dTR = H.'*dHTR;

        
        %reg4 = 0.5*delta4^2;
        ddelta4 = dreg*delta4;
        %delta4 = mdot(H) - D + d;
        dH = dH + (2*ddelta4)*H;
        
        
        %reg3 = mdot(HHHF,HF);
        dHHHF = dreg*HF;
        dHF = dreg*HHHF;
        %HHHF = HH*HF;
        dHH = dHHHF*HF.';
        dHF = dHF + HH.'*dHHHF;
        
        
        %reg2 = (trGG - 2*trGHH + trHHHH)/2
        dtrGG = dreg/2;
        dtrGHH = -dreg;
        dtrHHHH = dreg/2;
        
        
        %trHHHH = mdot(HH)
        dHH = dHH + (2*dtrHHHH)*HH;
        
        
        %trGHH = mdot(H) - mdot(HF,BFH.')
        dH = dH + (2*dtrGHH)*H;
        dHF = dHF - dtrGHH*BFH.';
        dBFH = (-dtrGHH)*HF.';
        
        %trGG = rdim  -  2*mdot(F,BF.') + mdot(FFB)
        dF = (-2*dtrGG)*BF.';
        dBF = (-2*dtrGG)*F.';
        dFFB = (2*dtrGG)*FFB;

        %HH = H*H.'   
        dH = dH + (2*dHH)*H;

        %BFH = BF*H.'
        dBF = dBF + dBFH*H;
        dH = dH + dBFH.'*BF;
        
        %HF = H*F
        dH = dH + dHF*F.';
        dF = dF + H.'*dHF;
        
        %FFB = F.'*BF.'
        dF = dF + BF.'*dFFB.';
        dBF = dBF + dFFB.'*F.';
        
        %BF = bsxfun(@ldivide,d,F.')
        dF = dF + bsxfun(@ldivide,d,dBF).';
        dd = dd - sum(bsxfun(@ldivide,d,BF.*dBF),2);
        
        %reg1 = 0.5*delta1^2;  
        ddelta1 = dreg*delta1;
        %delta1 = mdot(B0) - d.'*d;
        dB0 = (2*ddelta1)*B0;
        dd = dd - (2*ddelta1)*d;

        %d = diag(B0);
        dB0 = dB0 + diag(dd);
        
        %B0 = F.'*F;
        dF = dF + 2*F*dB0;  
        
        
        %A0 = F.'*TR;
        dF = dF + TR*dA0.';
        dTR = dTR + F*dA0;
        
        %TR = T*R;
        dT = dTR*R.';
        
        
        
        
    end
    
    

end

%trace(A*B.')
function y = mdot(A,B)
    if exist('B','var')
        assert(all(size(A)==size(B)));
        y = A(:).'*B(:);
    else
        y = sum(A(:).^2,1);
    end
end


function test_this()

    zdim = 2;
    rdim = 5;
    n = 4;
    F = randn(rdim,zdim);
    T = randn(rdim,rdim);
    H = randn(rdim-zdim,rdim);
    
    nu = pi;
    R = randn(rdim,n);
    
    f = @(T,F,H) SGME_extr(T,F,H,nu,R);
    
    testBackprop_multi(f,4,{T,F,H});

end
