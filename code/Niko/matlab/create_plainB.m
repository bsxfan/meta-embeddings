function B = create_plainB(B)

    cholBI = chol(I + B);
    B.BI.solve = @solve;
    
    function [mu,log_det] = solve(a)
        log_det = 2*sum(log(diag(cholBI)));
        mu = cholBI\(cholBI'\a);
    end

end