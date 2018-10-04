function [M,back] = matmul(A,B)

    if nargin==0
        test_this();
        return;
    end

    M = A*B;
    back = @back_this;
    
    function [dA,dB] = back_this(dM)
        dA = dM*B.';
        dB = A.'*dM;
    end


end

function test_this()

    m = 2;
    n = 3;
    k = 4;
    A = randn(m,k);
    B = randn(k,n);

    testBackprop(@matmul,{A,B});
    
    
end