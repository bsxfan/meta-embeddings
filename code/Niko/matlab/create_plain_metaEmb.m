function E = create_plain_metaEmb(a,B)


    E.log_expectation = @log_expectation;
    E.pool = @pool;
    E.getNatParams = @getNatParams;
    E.get_mu_cov = @get_mu_cov;
    E.scale = @scale;
    E.convolve = @convolve;
    
    
    function [a1,B1] = getNatParams()
        a1 = a;
        B1 = B;
    end
    
    function PE = pool(AE)
        [a1,B1] = AE.getNatParams();
        PE = create_plain_metaEmb(a+a1,B+B1);
    end

    % This is not scalar multiplication of the meta-embedding
    % It is scaling of the natural parameters.
    function PE = scale(s)
        PE = create_plain_metaEmb(s*a,s*B);
    end

    function y = log_expectation()
        dim = length(a);
        cholBI = chol(speye(dim) + B);
        log_det = 2*sum(log(diag(cholBI)));
        mu = cholBI\(cholBI'\a);
        y = (mu'*a - log_det)/2;
    end


    %for inspection purposes (eg plotting), not speed
    function [mu,C] = get_mu_cov()
        mu = B\a;
        C = inv(B);
    end
    

    function CE = convolve(AE)
        [mu1,C1] = AE.get_mu_cov();
        mu = mu1 + B\a;
        C = C1 + inv(B);
        CE = create_plain_metaEmb(C\mu,inv(C));
    end

    
end