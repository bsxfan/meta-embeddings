function [y,back] = mvn_obj(T,fi,params)

    if nargin==0
        test_this2();
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



function test_this2()


    dim = 5;
    n = 10000;
    rank = 2;
    %[f,fi,sz] = create_nice_Trans(dim,rank);
    %[f,fi,sz] = create_linTrans2(dim);
    [f,fi,sz] = create_affineTrans2(dim);
    R = randn(dim,n);
    oracle = randn(sz,1)/10;
    oracle(1) = pi;
    
    T = f(oracle,R);
    
    %params0 = randn(sz,1)/10;
    %params0(1) = 1/sqrt(sum(T(:).^2)/n);
    
    mu = mean(T,2);
    C = cov(T.',1);
    
    M = eye(dim)/sqrt(trace(C)/dim);
    params0 = [M(:);mu];
    
    %params0 = eye(dim)/sqrt(sum(T(:).^2)/n);
    %params0 = params0(:);
    
    obj = @(params) mvn_obj(T,fi,params);
    
    oracle_obj = obj(oracle),
    init_obj = obj(params0),
    
    mem = 20;
    stpsz0 = 1;
    
    [params,obj_final] = L_BFGS(obj,params0,100,2*60,20,1/100);    


    init_obj = obj(params0),
    oracle_obj = obj(oracle),
    final_obj = obj(params),
    
    
    
end
