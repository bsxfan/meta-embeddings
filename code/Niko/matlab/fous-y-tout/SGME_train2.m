function model = SGME_train2(R,labels,nu,zdim,reg_weight,niters,test)



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
    T0 = randn(rdim,rdim);
    F0 = randn(rdim,zdim);
    H0 = randn(rdim-zdim,rdim);
    
    szT = numel(T0);
    szF = numel(F0);
    szH = numel(H0);
    
    
    w0 = pack(T0,F0,H0);

    if exist('test','var') && test
        testBackprop(@objective,w0);
        return;
    end
    
    mem = 20;
    stpsz0 = 1e-3;
    timeout = 5*60;
    
    
    w = L_BFGS(@objective,w0,niters,timeout,mem,stpsz0);
    
    [T,F,H] = unpack(w);
    d = diag(F.'*F);
    
%     model.logexpectation = @(A,b) SGME_logexpectation(A,b,d);
%     model.extract = @(R) SGME_extr(T,F,nu,R);
%     model.objective = @(T,F) objective(pack(T,F));
%     model.d = d;
    model = create_HTPLDA_SGME_backend2(nu,T,F,H);


    function w = pack(T,F,H)
        w = [T(:);F(:);H(:)];
    end

    function [T,F,H] = unpack(w)
        at = 1:szT;
        T = reshape(w(at),rdim,rdim);
        at = szT + (1:szF);
        F = reshape(w(at),rdim,zdim);
        at = szT + szF + (1:szH);
        H = reshape(w(at),rdim-zdim,rdim);
    end
    
    
    
    
    function [y,back] = objective(w)
        
        [T,F,H] = unpack(w);
        
        [A,b,d,reg,back1] = SGME_extr(T,F,H,nu,R);
        
        [PsL,back2] = SGME_logPsL(A,b,d,blocks,labels,num,logPrior);
        y = reg_weight*reg - PsL;
        
        
        back = @back_this;
        
        function [dw] = back_this(dy)
            %dPsL = -dy;
            [dA,db,dd] = back2(-dy);
            [dT,dF,dH] = back1(dA,db,dd,reg_weight*dy);
            dw = pack(dT,dF,dH);
            
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

    reg_weight = exp(1);
    
    test = true;
    niters = [];
    SGME_train2(R,labels,nu,zdim,reg_weight,niters,test);


end

