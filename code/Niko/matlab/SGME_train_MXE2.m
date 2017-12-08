function model = SGME_train_MXE2(R,labels,nu,zdim,niters,timeout,test)



    if nargin==0
        test_this();
        return;
    end


    [rdim,n] = size(R);
    m = max(labels);
    blocks = sparse(labels,1:n,true,m,n);  
    counts = sum(blocks,2);
    logPrior = log(counts);
    
    
    
    delta = rdim - zdim;
    assert(delta>0);
    
    %initialize
    P0 = randn(zdim,rdim);
    H0 = randn(delta,rdim);
    sqrtd0 = rand(zdim,1);
    As0 = randn(zdim,m);
    sqrtBs0 = randn(1,m);
    
    szP = numel(P0);
    szH = numel(H0);
    szd = numel(sqrtd0);
    szAs = numel(As0);
    szBs = numel(sqrtBs0);
    
    
    w0 = pack(P0,H0,sqrtd0,As0,sqrtBs0);

    if exist('test','var') && test
        testBackprop(@objective,w0);
        return;
    end
    
    mem = 20;
    stpsz0 = 1e-3;
    %timeout = 5*60;
    
    
    w = L_BFGS(@objective,w0,niters,timeout,mem,stpsz0);
    
    [P,H,sqrtd,As,sqrtBs] = unpack(w);
    d = sqrtd.^2;
    
    model.logexpectation = @(A,b) SGME_logexpectation(A,b,d);
    model.extract = @(R) SGME_extract(P,H,nu,R);
    model.d = d;
    
    
    function w = pack(P,H,d,As,Bs)
        w = [P(:);H(:);d(:);As(:);Bs(:)];
    end

    function [P,H,d,As,Bs] = unpack(w)
        at = 1:szP;
        P = reshape(w(at),zdim,rdim);
        at = szP + (1:szH);
        H = reshape(w(at),delta,rdim);
        at = szP + szH + (1:szd);
        d = w(at);
        at = szP + szH + szd + (1:szAs);
        As = reshape(w(at),zdim,m);
        at = szP + szH + szd + szAs + (1:szBs);
        Bs = w(at).';
        
    end
    
    
    
    
    function [y,back] = objective(w)
        
        [P,H,sqrtd,As,sqrtBs] = unpack(w);
        
        [A,b,back1] = SGME_extract(P,H,nu,R);
        
        d = sqrtd.^2;
        Bs = sqrtBs.^2;
        
        [y,back2] = SGME_MXE2(A,b,d,As,Bs,labels,logPrior);
        
        
        back = @back_this;
        
        function [dw] = back_this(dy)
            [dA,db,dd,dAs,dBs] = back2(dy);
            dsqrtd = 2*sqrtd.*dd;
            dsqrtBs = 2*sqrtBs.*dBs;
            [dP,dH] = back1(dA,db);
            dw = pack(dP,dH,dsqrtd,dAs,dsqrtBs);
            
        end
        
        
    end






end

function test_this()

    zdim = 2;
    rdim = 4;
    n = 5;
    m = 3;
    
    prior =  create_PYCRP([],0,m,n);  
    labels = prior.sample(n);

    nu = pi;
    R = randn(rdim,n);

    test = true;
    niters = [];
    timeout = [];
    SGME_train_MXE2(R,labels,nu,zdim,niters,timeout,test);


end

