function [llr,PE,lognum,logden1,logden2] = metaEmb_llr(E1,E2)

    PE = E1.pool(E2);
    
    lognum = PE.log_expectation();
    logden1 = E1.log_expectation();
    logden2 = E2.log_expectation(); 
    
    llr = lognum - logden1 - logden2;
    

end