function [f,fi] = create_scalTrans()

    if nargout==0
        test_this();
        return;
    end


    f = @(scal,R) scal*R;
    
    fi = @fi_this;
    
    function [R,logdetJ,back] = fi_this(scal,T)
        [dim,N] = size(T);
        R = T/scal;
        logdetJ = (N*dim/2)*log(scal^2);
        back = @back_this;
    

        function dscal = back_this(dR,dlogdetJ)
            dscal = N*dim*dlogdetJ/scal;
            dscal = dscal - (R(:).'*dR(:))/scal;
        end
    
    end

end


function test_this()

    [f,fi] = create_scalTrans();
    R = randn(3,5);
    ff = @(scal) fi(scal,R);
    scal = pi;

    testBackprop_multi(ff,2,{scal});
    
    
end

