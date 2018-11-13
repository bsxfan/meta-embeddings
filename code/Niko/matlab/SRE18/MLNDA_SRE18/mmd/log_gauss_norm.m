function [y,back] = log_gauss_norm(B,a)

    if nargin==0
        test_this();
        return;
    end

    [logd,mu,back1] = logdet_solveLU(B,a);
    y = (logd -mu.'*a)/2;
    
    back = @back_this;
    
    function [dB,da] = back_this(dy)
        dlogd = dy/2;
        da = (-dy/2)*mu;
        dmu = (-dy/2)*a;
        [dB,da1] = back1(dlogd,dmu);
        da = da + da1;
    end
    
end

function test_this()

    dim = 5;
    B = randn(dim);
    a = rand(dim,1);
    
    testBackprop(@log_gauss_norm,{B,a},{1,1});


end
