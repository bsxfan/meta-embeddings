function [f,fi,paramsz,fe] = create_iso_lowrank_trans(dim,rank)

    if nargin==0
        test_this();
        return;
    end

    paramsz = 1 + 2*dim*rank + dim;
    [f0,fi0] = create_affineTrans(dim);

    f = @f_this;
    fi = @fi_this;
    fe = @expand;
    

    function T = f_this(P,R)
        Q = expand(P);
        T = f0(Q,R); 
    end
    

    function [R,logdetJ,back] = fi_this(P,T)
        [Q,back1] = expand(P);
        [R,logdetJ,back2] = fi0(Q,T);
        back = @back_that;
        
        function [dP,dT] = back_that(dR,dlogsetJ)
            if nargout==2
                [dQ,dT] = back2(dR,dlogsetJ);
            else
                dQ = back2(dR,dlogsetJ);
            end
            dP = back1(dQ);
        end
    end



    function [Q,back] = expand(P)
        at = 1;
        
        sz = 1;
        sigma = P(at);
        at = at + sz;
        
        sz = dim*rank;
        L = reshape(P(at:at+sz-1),dim,rank);
        at = at + sz;
        
        sz = dim*rank;
        R = reshape(P(at:at+sz-1),rank,dim);
        at = at + sz;
        
        sz = dim;
        offset = P(at:at+sz-1);
        at = at + sz;
        
        assert(at==length(P)+1);
        
        
        [LR,back1] = matmul(L,R);
        M = sigma*speye(dim) + LR;
        
        Q = [M(:);offset];
        
        
        
        back = @back_this;
        
        function dP = back_this(dQ)
            dQ = reshape(dQ,dim,dim+1);
            doffset = dQ(:,end);
            dM = dQ(:,1:end-1);
            dsigma = trace(dM);
            [dL,dR] = back1(dM);
            dP = [dsigma;dL(:);dR(:);doffset];
        end
        
        
        
        
    end



end


function test_this()

    dim = 5;
    rank = 2;
    n = 6;
    
    [f,fi,sz] = create_iso_lowrank_trans(dim,rank);
    P = randn(sz,1);
    R = randn(dim,n);
    T = f(P,R);
    
    
    
    testBackprop_multi(fi,2,{P,T});
    %testBackprop(fe,{P});
    
    
    


end

