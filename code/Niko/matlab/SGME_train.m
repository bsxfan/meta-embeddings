function model = SGME_train(R,labels,nu,zdim,niters,test)



    if nargin==0
        test_this();
        return;
    end


    [rdim,n] = size(R);
    m = max(labels);
    blocks = sparse(labels,1:n,true,m+1,n);  
    num = find(blocks(:));    
    
    %Can we choose maximum likelihood prior parameters, given labels?
    %For now: prior expected number of speakers = m
    prior =  create_PYCRP([],0,m,n);  
    logPrior = prior.GibbsMatrix(labels);
    
    
    
    delta = rdim - zdim;
    assert(delta>0);
    
    %initialize
    P0 = randn(zdim,rdim);
    H0 = randn(delta,rdim);
    sqrtd0 = rand(zdim,1);
    
    szP = numel(P0);
    szH = numel(H0);
    
    
    w0 = pack(P0,H0,sqrtd0);

    if exist('test','var') && test
        testBackprop(@objective,w0);
        return;
    end
    
    mem = 20;
    stpsz0 = 1e-3;
    timeout = 5*60;
    
    
    w = L_BFGS(@objective,w0,niters,timeout,mem,stpsz0);
    
    [P,H,sqrtd] = unpack(w);
    d = sqrtd.^2;
    
    model.logexpectation = @(A,b) SGME_logexpectation(A,b,d);
    model.extract = @(R) SGME_extract(P,H,nu,R);
    model.objective = @(P,H,d) objective(pack(P,H,d));
    model.d = d;
    
    
    function w = pack(P,H,d)
        w = [P(:);H(:);d(:)];
    end

    function [P,H,d] = unpack(w)
        at = 1:szP;
        P = reshape(w(at),zdim,rdim);
        at = szP + (1:szH);
        H = reshape(w(at),delta,rdim);
        at = szP + szH + (1:zdim);
        d = w(at);
        
    end
    
    
    
    
    function [y,back] = objective(w)
        
        [P,H,sqrtd] = unpack(w);
        
        [A,b,back1] = SGME_extract(P,H,nu,R);
        
        d = sqrtd.^2;
        
        [PsL,back2] = SGME_logPsL(A,b,d,blocks,labels,num,logPrior);
        y = -PsL;
        
        
        back = @back_this;
        
        function [dw] = back_this(dy)
            %dPsL = -dy;
            [dA,db,dd] = back2(-dy);
            dsqrtd = 2*sqrtd.*dd;
            [dP,dH] = back1(dA,db);
            dw = pack(dP,dH,dsqrtd);
            
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
    SGME_train(R,labels,nu,zdim,test);


end

