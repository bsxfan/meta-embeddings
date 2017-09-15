function E = create_plain_metaEmb(a,B)


    E.log_expectation = @log_expectation;
    E.add = @add;
    E.getNatParams = @getNatParams;
    E.get_mu_cov = @get_mu_cov;
    
    
    function [a1,B1] = getNatParams()
        a1 = a;
        B1 = B;
    end
    
    function PE = add(AE)
        [a1,B1] = AE.getNatParams();
        PE = create_plain_metaEmb(a+a1,B+B1);
    end

    function y = log_expectation()
        dim = length(a);
        cholBI = chol(speye(dim) + B);
        log_det = 2*sum(log(diag(cholBI)));
        mu = cholBI\(cholBI'\a);
        y = (mu'*a + log_det)/2;
    end


    %for inspection purposes (eg plotting), not speed
    function [mu,C] = get_mu_cov()
        mu = B\a;
        C = inv(B);
    end
    
    
end