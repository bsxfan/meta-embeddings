function [Y,back] = LinvSR(L,S,R)

    if nargin==0
        test_this();
        return;
    end

    
    Z = S\R;
    Y = L*Z;
    
    back = @back_this;
    
    function [dL,dS,dR] = back_this(dY)
        
        % Y = L*Z
        dL = dY*Z.';
        dZ = L.'*dY;

        % Z = S\R;
        dR = S.'\dZ;
        dS = -dR*Z.';
    end


end




function test_this()

    m = 3;
    n = 4;
    
    L = randn(m,n);
    R = randn(n,m);
    S = randn(n,n);
    
    fprintf('test slow derivatives:\n');
    testBackprop(@LinvSR,{L,S,R});
    
    
    
end

