function model = addME_train(R,labels,model,niters,timeout,test)



    if nargin==0
        test_this();
        return;
    end


    [~,n] = size(R);
    m = max(labels);
    blocks = sparse(labels,1:n,true,m+1,n);  
    num = find(blocks(:));    
    
    %Can we choose maximum likelihood prior parameters, given labels?
    %For now: prior expected number of speakers = m
    prior =  create_PYCRP([],0,m,n);  
    logPrior = prior.GibbsMatrix(labels);
    
    
    
    [w0_ext,w0_exp] = model.init();
    w0 = [w0_ext;w0_exp];
    sz1 = length(w0_ext);
    
    

    if exist('test','var') && test
        testBackprop(@objective,w0);
        return;
    end
    
    mem = 20;
    stpsz0 = 1e-3;
    
    if ~exist('timeout','var') || isempty(timeout)
        timeout = 5*60;  %5 min
    end
    
    
    w = L_BFGS(@objective,w0,niters,timeout,mem,stpsz0);
    
    [P,H,sqrtd] = unpack(w);
    d = sqrtd.^2;
    
    model.logexpectation = @(A,b) SGME_logexpectation(A,b,d);
    model.extract = @(R) SGME_extract(P,H,nu,R);
    model.objective = @(P,H,d) objective(pack(P,H,d));
    model.d = d;
    
    

    function [w_ext,w_exp] = unpack(w)
        w_ext = w(1:sz1);
        w_exp = w(sz1+1:end);
        
    end
    
    
    
    
    function [y,back] = objective(w)
        
        [w_ext,w_exp] = unpack(w);
        
        [E,back1] = model.extract(w_ext,R);
        
        
        [y,back2] = addME_pseudoscore(E,w_exp,model.logexpectations,blocks,labels,num,logPrior);
        
        
        back = @back_this;
        
        function dw = back_this(dy)
            [dE,dw_exp] = back2(dy);
            dw_ext = back1(dE);
            dw = [dw_ext;dw_exp];
            
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
    SGME_train(R,labels,nu,zdim,niters,test);


end

