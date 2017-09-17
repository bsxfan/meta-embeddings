function [llr,f1f2,lognum,logden1,logden2] = metaEmb_llr(f1,f2)
% Computes logLR for two meta-embeddings, f1 and f2
%
%   log LR = log E{f1f2} - log E{f1} - log E{f2}
    
    f1f2 = f1.pool(f2);
    
    lognum = f1f2.log_expectation();
    logden1 = f1.log_expectation();
    logden2 = f2.log_expectation(); 
    
    llr = lognum - logden1 - logden2;
    

end