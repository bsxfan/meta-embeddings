function node = create_symmetricDirichlet_node(logalpha,sz)
% Creates symmetric Dirichlet node (part of a Bayesian network), that is
% equippped to do alternating Gibbs sampling. This node expects a
% non-conjugate (upstream) hyper-prior to supply the Dirichlet parameter
% alpha. It also expects (downstream) one or more categorical
% distributions (conjugate), all parametrized IID by the vector w, 
% where w = {w_i} are the category probabilities.
%
% Inputs:
%   sz: the number of categories
%   
% Output:
%   node.sample: function handle to do a Gibbs sampling step, sampling from
%                P(w | alpha, counts). The function returns messages to all 
%                its neighbours: 
%                    log(w): downstream to the conjugate categorical 
%                            children.
%                    likelhood function for log alpha: log P(w | log alpha) 
%                                                      to the non-conjugate 
%                                                      upstream parent. 



    if nargin==0
        test_this();
        return;
    end

    alpha = exp(logalpha);
    counts = 0;
    w = randg(alpha+counts,sz,1);
    w = w/sum(w);
    logw = log(w);
    
     
    
    node.get = @get;
    node.sample = @sample;
    node.condition_on_child = @condition_on_child;
    node.condition_on_parent = @condition_on_parent;
    node.inferParent = @inferParent;
    node.logPosterior_at_default = @logPosterior_at_default;
    
    
    function logP = logPosterior_at_default()
        % default: w = 1/sz
        alpha_c = alpha + counts;
        total = sum(alpha_c);
        logP = (sz-total)*log(sz) + gammaln(total) - sum(gammaln(alpha_c)); 
    end
    
    
    function val = get()
        val = logw;
    end
    
    function condition_on_parent(logalpha_new)
        logalpha = logalpha_new;
        alpha = exp(logalpha);
    end
    
    function condition_on_child(observed_counts)
        counts = observed_counts(:);
    end
        

    function val = sample(n)
        if nargin==0
            n = 1;
        end
        
        w = randg(alpha+counts,sz,n);
        w = w/sum(w);
        val = log(w);

        if nargin==0
            logw = val;
        end
    
    end
    

    % Outputs:
    %   ncm: non-conjugate-message (likelihood function) sent to hyper-prior for alpha
    function ncm = inferParent(givenChild)     
        if nargin==0 || ~givenChild   %infer parent given the current value of this node
            sumlogw = sum(logw);
            ncm.llh = @(logalpha) llh_this(logalpha,sz,sumlogw);
            ncm.Hessian = @(logalpha) llh_Hessian(logalpha,sz,sumlogw);
        else %infer parent after collapsing this node
            ncm.llh = @(logalpha) llh_collapsed(logalpha,counts);
            ncm.Hessian = @(logalpha) collapsed_Hessian(logalpha,counts);
        end
    end
        
        
    
end



function [y,y1] = llh_collapsed(logalpha,counts)
    alpha = exp(logalpha);
    sz = length(counts);
    sumC = full(sum(counts));
    y = gammaln(sz*alpha) - gammaln(sz*alpha+sumC) + ...
        sum(gammaln(bsxfun(@plus,alpha,counts)),1) - sz*gammaln(alpha); 
    if nargout>1
        y1 = alpha.*( sz*psi(sz*alpha) - sz*psi(sz*alpha+sumC) + ...
                      sum(psi(bsxfun(@plus,alpha,counts)),1) - sz*psi(alpha) );
    end
end


function h = collapsed_Hessian(logalpha,counts)
    alpha = exp(logalpha);
    sz = length(counts);
    sumC = full(sum(counts));
    y1 = alpha.*( sz*psi(sz*alpha) - sz*psi(sz*alpha+sumC) + ...
                  sum(psi(bsxfun(@plus,alpha,counts)),1) - sz*psi(alpha) );
    h = y1 + alpha.^2.*( sz^2*psi(1,sz*alpha) - sz^2*psi(1,sz*alpha+sumC) + ...
                         sum(psi(1,bsxfun(@plus,alpha,counts)),1) - sz*psi(1,alpha) );
end




function [y,y1] = llh_this(logalpha,sz,sumlogw)
    alpha = exp(logalpha);
    y = expm1(logalpha)*sumlogw + gammaln(sz*alpha) - sz*gammaln(alpha); 
    if nargout>1
        y1 = alpha.*(sumlogw + sz*psi(sz*alpha) - sz*psi(alpha) );
    end
end


function h = llh_Hessian(logalpha,sz,sumlogw)
    alpha = exp(logalpha);
    y1 = alpha.*(sumlogw + sz*psi(sz*alpha) - sz*psi(alpha) );
    h = y1 + alpha.*(sz^2*alpha.*psi(1,sz*alpha) - sz*alpha.*psi(1,alpha) );
end
    



function test_this()

    close all;
    
    sz = 100;
    n = 1000;
    logalpha0 = 3;
    
    wnode = create_symmetricDirichlet_node(logalpha0,sz);
    logw0 = wnode.get();
    labelnode = create_Label_node(logw0,n);
    counts = labelnode.inferParent();
    wnode.condition_on_child(counts);
    wnode.sample();
    
    ncm = wnode.inferParent();
    logalpha = -5:0.01:5;
    [y,y1] = ncm.llh(logalpha);
    delta = 1e-3;
    [yplus,y1plus] = ncm.llh(logalpha+delta);
    [ymin,y1min] = ncm.llh(logalpha-delta);
    rstep1 = (yplus-ymin)/(2*delta);
    rstep2 = (y1plus-y1min)/(2*delta);
    h = ncm.Hessian(logalpha);
    subplot(2,2,1);plot(logalpha,y,logalpha,y1,logalpha,rstep1,'--',...
                        logalpha,h,logalpha,rstep2,'--');legend('y','y1','rstep1','hess','rstep2');grid;
    subplot(2,2,3);plot(logalpha,exp(y-max(y)),logalpha0,0,'g*');grid;
    
    
    ncm = wnode.inferParent(true);
    logalpha = -5:0.01:10;
    [y,y1] = ncm.llh(logalpha);
    delta = 1e-3;
    [yplus,y1plus] = ncm.llh(logalpha+delta);
    [ymin,y1min] = ncm.llh(logalpha-delta);
    rstep1 = (yplus-ymin)/(2*delta);
    rstep2 = (y1plus-y1min)/(2*delta);
    h = ncm.Hessian(logalpha);
    subplot(2,2,2);plot(logalpha,y,logalpha,y1,logalpha,rstep1,'--',...
                        logalpha,h,logalpha,rstep2,'--');legend('y','y1','rstep1','hess','rstep2');grid;
    subplot(2,2,4);plot(logalpha,exp(y-max(y)),logalpha0,0,'g*');grid;
    

    
    
    
    
    
end


