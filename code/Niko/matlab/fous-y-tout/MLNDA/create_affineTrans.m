function [f,fi] = create_affineTrans()

    if nargout==0
        test_this();
        return;
    end


    f = @(offset,M,R) bsxfun(@plus,offset,M*R);
    
    fi = @fi_this;
    
    function [R,logdetJ,back] = fi_this(offset,M,T)

        [L,U] = lu(M);
        n = size(T,2);
        logdetJ = n*sum(log(diag(U).^2))/2;
        Delta = bsxfun(@minus,T,offset);
        R = U\(L\Delta);
        back = @back_this;
    

        function [doffset,dM] = back_this(dR,dlogdetJ)
            dM = (n*dlogdetJ)*(U\inv(L)).';
            dDelta = L.'\(U.'\dR);
            dM = dM - dDelta*R.';
            doffset = -sum(dDelta,2);
        end
    
    end

end


function test_this()

    [f,fi] = create_affineTrans();
    R = randn(3,5);
    ff = @(offset,M) fi(offset,M,R);
    offset = randn(3,1);
    M = randn(3);

    testBackprop_multi(ff,2,{offset,M},{1,1});
    
    
end
