function [f,fi,paramsz] = create_diaglinTrans(dim)

    if nargout==0
        test_this();
        return;
    end


    f = @f_this; 
    
    fi = @fi_this;
    
    paramsz = dim;
    
    function T = f_this(P,R)
        T = bsxfun(@times,P,R);
    end
    
    
    function [R,logdetJ,back] = fi_this(P,T)
        
        n = size(T,2);
        logdetJ = n*sum(log(P.^2))/2;
        R = bsxfun(@ldivide,P,T);
        back = @back_this;
    

        function [dP,dT] = back_this(dR,dlogdetJ)
            dP = n*dlogdetJ./P;
            dT = bsxfun(@ldivide,P,dR);
            %dP = dP - diag(dT*R.');
            dP = dP - sum(dT.*R,2);
        end
    
    end



end

function test_this()

    dim = 3;
    [f,fi,sz] = create_diaglinTrans(dim);
    R = randn(dim,5);
    P = randn(sz,1);
    T = f(P,R);

    Ri = fi(P,T);
    test_inverse = max(abs(R(:)-Ri(:))),

    testBackprop_multi(fi,2,{P,T});
    
    
end

