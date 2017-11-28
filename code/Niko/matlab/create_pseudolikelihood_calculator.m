function calc = create_pseudolikelihood_calculator(log_expectations,prior,poi)
% Inputs:
%   log_expectations: function handle
%   poi: partition of interest, n-vector that maps customers to tables
%   prior: exchangeble prior, struct, for example CRP
%

    n = length(poi);  % # customers
    m = max(poi);     % # tables

    labels = poi;
    
    %m-by-n index matrix, with one-hot columns
    blocks = sparse(poi,1:n,true,m,n);  %add extra row for empty table
    %LLR = zeros(m+1,n);

    logPrior = prior.GibbsMatrix(poi);  
    %logPrior = prior.slowGibbsMatrix(poi);
    
    
    
    calc.log_pseudo_likelihood = @log_pseudo_likelihood;
    calc.slow_log_pseudo_likelihood = @slow_log_pseudo_likelihood;
    
    function y = log_pseudo_likelihood(A,B)
        
        %[dimA,nA] = size(A);assert(n==nA);
        %[dimB,nB] = size(B);assert(n==nB);
        
        %original table stats
        At = A*blocks.';
        Bt = B*blocks.';
        %log-expectations for every original table
        LEt = log_expectations(At,Bt);
        
        %log-expectations for every customer, alone at singleton table
        LEc = log_expectations(A,B);
        
        %for every customer: log-expectation for the rest of the table,
        %                    excluding this customer
        LEmin = log_expectations(At(:,labels) - A,Bt(:,labels) - B);
        
        for i=1:m
             tar = blocks(i,:);  %target trials in this row 
             non = ~tar;         %the non-targets 
             
             LLR(i,tar) = LEt(labels(tar)) - LEc(tar) - LEmin(tar);
             
             Aplus = At(:,labels(non)) + A(:,non);
             Bplus = Bt(:,labels(non)) + B(:,non);
             LLR(i,non) = log_expectations(Aplus,Bplus) - LEc(non) - LEt(i);
        end
        %LLR(m+1,:) = 0; already zeroed
        
        logPost = LLR + logPrior;
        M = max(logPost,[],1);
        Den = M + log(sum(exp(bsxfun(@minus,logPost,M)),1));
        y = sum(logPost(blocks)) - sum(Den); 
    
    end

    function y = slow_log_pseudo_likelihood(A,B)
        
        %[dimA,nA] = size(A);assert(n==nA);
        %[dimB,nB] = size(B);assert(n==nB);
        
        %original table stats
        At = A*blocks.';
        Bt = B*blocks.';
        %log-expectations for every original table
        LEt = log_expectations(At,Bt);
        
        %log-expectations for every customer, alone at singleton table
        LEc = log_expectations(A,B);  
        
        %for every customer: log-expectation for the rest of the table,
        %                    excluding this customer
        %LEmin = log_expectations(At(:,labels) - A,Bt(:,labels) - B);
        
        y = 0;
        LLR = zeros(m+1,1);
        for j=1:n
            tj = labels(j); 
            Aplus = bsxfun(@plus,A(:,j),At);
            Bplus = bsxfun(@plus,B(:,j),Bt);
            LLR(1:m) = log_expectations(Aplus,Bplus).' - LEt.' - LEc(j);
            
            Amin = At(:,tj) - A(:,j);
            Bmin = Bt(:,tj) - B(:,j);
            LLR(tj) = LEt(tj) - log_expectations(Amin,Bmin) - LEc(j);
            
            
            LLR(m+1) = 0; 
            
            logPost = LLR + logPrior(:,j);
            M = max(logPost);
            Den = M + log(sum(exp(logPost-M)));
            y = y + logPost(tj) - Den; 
            post = exp(logPost - Den);
            
        end
        
    
    end










end