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
    
    
    
    [w0_ext,w0_exp] = model.getParams();
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
    
    model.getParams = @() unpack(w);    

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
    delta = rdim - zdim;
    
    n = 5;
    m = 3;
    
    prior =  create_PYCRP([],0,m,n);  
    labels = prior.sample(n);

    R = randn(rdim,n);

    test = true;
    niters = [];
    timeout = [];
    
    function [w_ext,w_exp] = init1()
        P = randn(zdim,rdim);
        H = randn(delta,rdim);
        w_ext = [P(:);H(:)];
        w_exp = randn(zdim,1);
    end
    
    function [w_ext,w_exp] = init2()
        nu = pi;
        P = randn(zdim,rdim);
        H = randn(delta,rdim);
        w_ext = [nu;P(:);H(:)];
        w_exp = randn(zdim,1);
    end
    
    learn_nu = true;


    if learn_nu
        model.getParams = @init2;
        model.extract = @(w,R) SGME_extract(w,zdim,R);
    else
        model.getParams = @init1;
        model.extract = @(w,R) SGME_extract(w,zdim,R,pi);
    end
    model.logexpectations = @SGME_logexpectations;


    
    addME_train(R,labels,model,niters,timeout,test);


end

