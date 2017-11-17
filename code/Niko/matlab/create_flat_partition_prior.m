function prior = create_flat_partition_prior(n)

    if n>100
        const = - approx_log_Bell(n);
    else
        const = - logBell(n);
    end
    
    
    prior.logProb = @(counts) const;


end