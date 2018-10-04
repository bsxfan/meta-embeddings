function [f,fi,paramsz] = create_shiftTrans(dim)

    if nargout==0
        test_this();
        return;
    end


    f = @f_this; 
    
    fi = @fi_this;
    
    paramsz = dim;
    
    function T = f_this(P,R)
        T = bsxfun(@plus,P,R);
    end
    
    
    function [R,logdetJ,back] = fi_this(P,T)
        
        logdetJ = 0;
        R = bsxfun(@minus,T,P);
        back = @back_this;
    

        function [dP,dT] = back_this(dR,dlogdetJ)
            dP = -sum(dR,2);
            dT = dR;
        end
    
    end



end

function test_this()

    dim = 3;
    [f,fi,sz] = create_shiftTrans(dim);
    P = randn(sz,1);
    R = randn(dim,5);
    T = f(P,R);

    testBackprop_multi(fi,2,{P,T});
    
    
end

