function calc = create_pseudolikelihood_calculator(log_expectations,prior,poi)
% Inputs:
%   log_expectations: function handle
%   poi: partition of interest, n-vector that maps customers to tables
%   prior: exchangeble prior, struct, for example CRP
%

    n = length(poi);  % # customers
    m = max(poi);     % # tables

    
    %(m+1)-by-n index matrix, with one-hot columns
    blocks = sparse(poi,1:n,true,m+1,n);  
    num = find(blocks(:));    
    
    
    %pre-allocate
    LLR = zeros(m+1,n);

    logPrior = prior.GibbsMatrix(poi);  
    %logPrior = prior.slowGibbsMatrix(poi);
    
    
    
    calc.log_pseudo_likelihood = @log_pseudo_likelihood;
    
    function [y,back] = log_pseudo_likelihood(A,B)
        %[dimA,nA] = size(A);assert(n==nA);
        %[dimB,nB] = size(B);assert(n==nB);
        
        %original table stats
        At = A*blocks.';
        Bt = B*blocks.';
        %log-expectations for every original table
        LEt = log_expectations(At,Bt);
        
        %log-expectations for every customer, alone at singleton table
        LEc = log_expectations(A,B);  

        
        for i=1:m
            tar = full(blocks(i,:));
            non = ~tar;
            
            %non-targets
            Aplus = bsxfun(@plus,A(:,non),At(:,i));
            Bplus = bsxfun(@plus,B(:,non),Bt(:,i));
            LLR(i,non) = log_expectations(Aplus,Bplus) - LEt(i) - LEc(non);
            
            %targets
            Amin = bsxfun(@minus,At(:,i),A(:,tar));
            Bmin = bsxfun(@minus,Bt(:,i),B(:,tar));
            LLR(i,tar) = LEt(i) - log_expectations(Amin,Bmin) - LEc(tar);
            
        end

        logPost = LLR + logPrior;
        M = max(logPost,[],1);
        Den = M + log(sum(exp(bsxfun(@minus,logPost,M)),1));
        y = sum(logPost(num),1) - sum(Den,2);    
        
        
        back = @back_this;
        
        
        function [dA,dB] = back_this()
            
        end
        
        
    
    end








end