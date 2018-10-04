function [f,fi,paramsz,fe] = create_sandwich_trans(dim,rank)

    if nargin==0
        test_this();
        return;
    end

    [f1,fi1] = create_affineTrans(dim);
    [f2,fi2] = create_diaglinTrans(dim);
    [f3,fi3] = create_linTrans2(dim);

    paramsz = 1 + 2*dim*rank + 2*dim;

    f = @f_this;
    fi = @fi_this;
    fe = @expand;
    

    function T = f_this(P,R)
        [A,D,L] = expand(P);
        T = f1(A,f2(D,f3(L,R))); 
    end
    

    function [R,logdetJ,back] = fi_this(P,T)
        [A,D,L,back4] = expand(P);
        [T1,logdetJ1,back1] = fi1(A,T);
        [T2,logdetJ2,back2] = fi2(D,T1);
        [R,logdetJ3,back3] = fi3(L,T2);
        logdetJ = logdetJ1 + logdetJ2 + logdetJ3;
        back = @back_that;
        
        function [dP,dT] = back_that(dR,dlogdetJ)
            [dL,dT2] = back3(dR,dlogdetJ);
            [dD,dT1] = back2(dT2,dlogdetJ);
            if nargout==2
                [dA,dT] = back1(dT1,dlogdetJ);
            else
                dA = back1(dT1,dlogdetJ);
            end
            dP = back4(dA,dD,dL);
        end
    end



    function [A,D,L,back] = expand(P)
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
        
        sz = dim;
        D = P(at:at+sz-1);
        at = at + sz;

        assert(at==length(P)+1);
        
        
        [LR,back1] = matmul(L,R);
        L = sigma*speye(dim) + LR;
        
        L = L(:);
        A = [L;offset];
        
        
        
        back = @back_this;
        
        function dP = back_this(dA,dD,dL)
            dA = reshape(dA,dim,dim+1);
            dL = reshape(dL,dim,dim);
            doffset = dA(:,end);
            dM = dA(:,1:end-1) + dL;
            dsigma = trace(dM);
            [dL,dR] = back1(dM);
            dP = [dsigma;dL(:);dR(:);doffset;dD];
        end
        
        
        
        
    end



end


function test_this()

    dim = 5;
    rank = 2;
    n = 6;
    
    [f,fi,sz,fe] = create_sandwich_trans(dim,rank);
    P = randn(sz,1);
    R = randn(dim,n);
    T = f(P,R);
    
    Ri = fi(P,T);
    test_inverse = max(abs(R(:)-Ri(:))),
    
    
    testBackprop_multi(fi,2,{P,T});
    %testBackprop_multi(fe,3,{P});
    
    
    


end

