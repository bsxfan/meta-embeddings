function calc = create_pseudolikelihood_calculator(log_expectations,prior,poi)

    n = length(poi);
    m = max(poi);

    %m-by-n index matrix, with one-hot columns
    blocks = sparse(poi,1:n,true,m+1,n);  %add extra row for empty table
    LLR = zeros(m+1,n);

    logPrior = prior.Gibbs(labels);
    
    
    
    calc.log_pseudo_likelihood = @log_pseudo_likelihood;
    
    function y = log_pseudo_likelihood(A,B)
        
        %[dimA,nA] = size(A);assert(n==nA);
        %[dimB,nB] = size(B);assert(n==nB);
        
        %original table stats
        At = A*blocks.';
        Bt = B*blocks.';
        %log-expectations for every original table
        LEt = log_expectations(At,Bt);
        
        %log-expectations for every customer at his own table
        LEc = log_expectations(A,B);
        
        %for every customer: log-expectation for the rest of the table,
        %                    excluding this customer
        LEmin = log_expectations(At(:,labels) - A,Bt(:,labels) - B);
        
        for i=1:m
             tar = blocks(i,:);  %target trials in this row 
             non = ~tar;         %the non-targets 
             
             LLR(i,tar) = LEt(labels(tar)) - LEc(non) - LEmin(tar);
             
             Aplus = At(labels(non)) + A(non);
             Bplus = Bt(labels(non)) + B(non);
             LLR(i,non) = log_expectations(Aplus,Bplus) - LEc(non) - LEt(i);
        end
        %LLR(m+1,:) = 0; already zeroed
        
        logPost = LLR + logPrior;
        M = max(logPost,[],1);
        Den = M + log(sum(exp(bsxfun(@minus,logPost,M)),1));
        y = sum(logPost(blocks)) - sum(Den); 
    
    end











end