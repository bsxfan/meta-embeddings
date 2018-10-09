function [y,mu,back] = logdet_solveLU(B,a)

    if nargin==0
        test_this();
        return;
    end

    [L,U] = lu(B);
    y = sum(log(diag(U).^2))/2;  %logdet
    mu = U\(L\a);  %solve
    
    back = @back_this;
    
    function [dB,da] = back_this(dy,dmu)
        dB = dy*(L.'\inv(U.'));    
        da = L.'\(U.'\dmu);
        dB = dB - da*mu.';
    end


end

function test_this()

    dim = 5;
    B = randn(dim);
    a = randn(dim,1);
    
    testBackprop_multi(@logdet_solveLU,2,{B,a});

end