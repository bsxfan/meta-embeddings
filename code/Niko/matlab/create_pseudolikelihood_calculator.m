function calc = create_pseudolikelihood_calculator(log_expectations,prior,poi)

    n = length(poi);
    m = max(poi);

    blocks = sparse(1:n,poi,true,n,m+1,n);  %add extra empty table
    
    
    %precompute logprior(rest of labels when label i removed)
    logprior_rest = zeros(1,n);
    ci = counts;
    for k=1:n
        ti = labels(k);
        ci(ti) = ci(ti) - 1;
        logprior_rest(k) = prior.logprob(ci(ci>0));     
        ci(ti) = counts(ti);
    end
    
    
    calc.log_pseudo_likelihood = @log_pseudo_likelihood;
    
    function y = log_pseudo_likelihood(A,B)
        
        %[dimA,nA] = size(A);assert(n==nA);
        %[dimB,nB] = size(B);assert(n==nB);
        
        %original table stats
        At = A*blocks;
        Bt = B*blocks;
        %log-expectations for every original table
        LEt = log_expectations(At,Bt);
        
        %for every customer: log-expectation for the rest of the table,
        %                    excluding this customer
        LEmin = log_expectations(At(:,labels) - A,Bt(:,labels) - B);
        
        
        %log-expectations for every table, (re)joined by every customer
        LEx = zeros(n,m);
        for j=1:m+1
            bj = blocks(:,j).'; %customers at this table
            Aplus = bsxfun(@plus,At(:,j),A); %table is joined by every customer 
            Aplus(bj) = Aplus(bj) - A(bj);   %except those already sitting there
            Bplus = bsxfun(@plus,Bt(:,j),B);
            Bplus(bj) = Bplus(bj) - B(bj);
            LEx(:,j) = log_expectations(Aplus,Bplus).';
        end
        
        %log-expectations for every customer sitting at his own new table
        LEc = LEx(:,m+1);
        
        y = 0;
        for i=1:n
            left = LEc(i);
            right = LEt;
            right(labels(i)) = LEmin(i);
            both = LEx(:,i).'; 
            LLR = both - left - right;
        end

        LLR = bsxfun(@minus,LEx,LEt);
        LLR = bsxfun(@minus,LLR,LEc);
        LLR = LLR + LEt() - LEmin(); 
        
    
    end











end