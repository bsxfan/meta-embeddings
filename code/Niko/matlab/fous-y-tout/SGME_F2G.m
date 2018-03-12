function [G,reg,back] = SGME_F2G(F)


    if nargin==0
        test_this();
        return;
    end

    
    B0 = F.'*F;
    D = diag(B0);
    
    
    



end


function [y,back] = regGG(G)
    dim = size(G,1); 
    Delta = G*G-G;
    [y,back1] = regL2(Delta);
    back = @back_this;
    function [dG] = back_this(dy)
        dDelta = back1(dy);
        dG = G.'*dDelta + dDelta*G.' - speye(dim);
    end
    
end

function [y,back] = regDiag(B0,D)
    Delta = B0-D;
    [y,back1] = regL2(Delta);
    back = @back_this;
    function [dB0,dD] = back_this(dy)
        dDelta = back1(dy);
        dB0 = dDelta;
        dD = -dDelta;
    end
end

function [y,back] = regGF(G,F)
    Delta = G*F;
    [y,back1] = regL2(Delta);
    back = @back_this;
    function [dG,dF] = back_this(dy)
        dDelta = back1(dy);
        dG = dDelta*F.';
        dF = G.'*dDelta;
    end
end


function [y,back] = regL2(Delta)
    y = (Delta(:).'*Delta(:))/2;
    back = @(dy) Delta;
end