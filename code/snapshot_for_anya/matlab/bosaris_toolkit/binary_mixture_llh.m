function llh = binary_mixture_llh(llh1,llh2,logitp)
    logp = -log1p(exp(-logitp));
    logp1 = -log1p(exp(logitp));
    llh = logsumexp([logp + llh1; logp1 + llh2]);
    
end