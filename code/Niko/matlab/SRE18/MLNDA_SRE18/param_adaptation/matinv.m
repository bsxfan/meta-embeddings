function [Y,back] = matinv(S)

    if nargin==0
        test_this();
        return;
    end

    
    Y = inv(S);
    
    back = @back_this;
    
    function dS = back_this(dY)
        
        dS = -Y.'*dY*Y.';
    end


end




function test_this()

    n = 4;
    
    S = randn(n,n);
    
    testBackprop(@matinv,S);
    
    
    
end

