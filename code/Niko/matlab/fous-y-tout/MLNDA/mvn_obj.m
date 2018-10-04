function [y,back] = mvn_obj(T,fi,params)

    if nargin==0
        test_this();
        return;
    end


    [R,logdetJ,back2] = fi(params,T);
    [llh,back1] = smvn_llh(R);
    y = logdetJ - llh;
    
    back = @back_this;
    
    function dparams = back_this(dy)
        dlogdetJ = dy;
        dR = back1(-dy);
        dparams = back2(dR,dlogdetJ);
    end


end

function test_this()

    dim = 5;
    n = 20;
    rank = 2;
    R = randn(dim,n);
    
    [f,fi,sz] = create_nice_Trans(dim,rank);
    params = randn(sz,1);
    
    T = f(params,R);
    Ri = fi(params,T);
    test_inverse = max(abs(Ri(:)-R(:))),
    
    
    testBackprop(@(params)mvn_obj(T,fi,params),{params});
    


end