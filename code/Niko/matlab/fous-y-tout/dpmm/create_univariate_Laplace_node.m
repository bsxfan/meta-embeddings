function node = create_univariate_Laplace_node(prior)

    if nargin==0
        test_this();
        return;
    end

    value = prior.sample();
    
    %posterior stuff
    post_mu = []; %approximate posterior mean
    post_h = []; %approximate posterior hessian (- precision)
    %logpost = []; %unnormalized true log posterior
    
    
    node.get = @get;
    node.sample = @sample;
    node.condition_on_child = @condition_on_child;
    node.logPosterior_at_mode = @logPosterior_at_mode;
    
    function val = get()
        val = value;
    end
    

    % samples from prior/posterior
    % if n is given, draws n samples and does not change internal value
    % if n is not given, draws one sample and does change internal value
    function val = sample(n)
        if nargin==0
            n = 1;
        end
        if isempty(post_mu)
            val = prior.sample(n);
        else
            val = post_mu + randn(1,n)/sqrt(-post_h);
        end
        if nargin==0
            value = val;
        end
    end


    function [logPost,mode] = logPosterior_at_mode()
        if isempty(post_mu)
            [logPost,mode] = prior.logPosterior_at_mode();
        else
            mode = post_mu;
            logPost = log(-post_h)/2;
        end
    end


    function [mu,sigma,logpost] = condition_on_child(llh_message)
        s0 = value;
        f = @(s) - llh_message.llh(s) - prior.logPDF(s); 
        [mu,fmin,flag] = fminsearch(f,s0);
        assert(flag==1,'fminsearch failed');
        h = llh_message.Hessian(mu) + prior.Hessian(mu);
        
        post_mu = mu;
        post_h = h;
        sigma = 1/sqrt(-h);
        
        if nargout>=3
            logpost = @(s) fmin - f(s);
        end
    end


end

function test_this()

    close all;
    
    mu = 0;
    v = 5;
    wsz = 100;

    prior = create_uvg_Prior(mu,v);
    alpha_node = create_univariate_Laplace_node(prior);
    logalpha0 = alpha_node.get();
    
    w_node = create_symmetricDirichlet_node(logalpha0,wsz);
    %logw0 = w_node.get();
    
    ncm = w_node.inferParent();
    
    [mu,sigma,logpost] = alpha_node.condition_on_child(ncm);
    x = (-1:0.01:1)*4*sigma + mu;
    y = exp(logpost(x));
    lap = exp((x-mu).^2/(-2*sigma^2));
    
    logalpha = alpha_node.sample();
    
    
    plot(x,y,'g',x,lap,'r--',logalpha,0,'r*',logalpha0,0,'g*');grid;
    legend('posterior','laplace','laplace sample','true value');
    
end