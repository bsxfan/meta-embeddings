function calc = create_pseudolikelihood_calculator(log_expectations,prior,poi)

    n = length(poi);
    m = max(poi);
    [subsets,counts] = labels2blocks(poi);
    subsets = sparse(subsets);
    
    %precompute logprior(rest of labels when label i removed)
    logprior_rest = zeros(1,n);
    ci = counts;
    for i=1:n
        ti = labels(i);
        ci(ti) = ci(ti) - 1;
        logprior_rest(i) = prior.logprob(ci(ci>0));     
        ci(ti) = counts(ti);
    end
    
    function y = log_pseudo_likelihood(A,B)
        log_ex1 = log_expectations(A,B);
        log_ex_poi = log_expectations(A*subsets,B*subsets);
    end











end