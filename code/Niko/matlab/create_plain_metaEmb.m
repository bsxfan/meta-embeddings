function E = create_plain_metaEmb(a,B)
% Creates an object representing a multivariate Gaussian meta-embedding.
% The Gaussian is represented by its natural parameters, a in R^d 
% and B, a positve semi-definte matrix. The meta-embedding is: 
%   f(z) = exp(a'z -(1/2)z'Bz)
%
% The `object' created here is home-made in the sense that it does not use
% MATLAB's object-oriented mechanisms. Rather it is a struct, containing
% several function handles, all of which have access to a common set of 
% persistent, encapsulated local variables (here just a,B).

    E.log_expectation = @log_expectation;
    E.pool = @pool;
    E.getNatParams = @getNatParams;
    E.get_mu_cov = @get_mu_cov;
    E.scale = @scale;
    E.convolve = @convolve;
    
    
    % returns the same a,B used in construction
    function [a1,B1] = getNatParams()
        a1 = a;
        B1 = B;
    end

    % Returns new object, constructed with sum of natural parameters of 
    % this Gaussian and another represented by AE. This is just the product 
    % of the two Gaussians.
    function PE = pool(AE)
        [a1,B1] = AE.getNatParams();
        PE = create_plain_metaEmb(a+a1,B+B1);
    end

    % This is not scalar multiplication of the meta-embedding
    % It is scaling of the natural parameters.
    function PE = scale(s)
        PE = create_plain_metaEmb(s*a,s*B);
    end

    % Computes log E{f(z)}, w.r.t. N(0,I)
    function y = log_expectation()
        dim = length(a);
        cholBI = chol(speye(dim) + B);
        logdetBI = 2*sum(log(diag(cholBI)));
        %mu = cholBI\(cholBI'\a);
        %y = (mu'*a - log_det)/2;
        z = cholBI.'\a;
        y = (z.'*z - logdetBI)/2;
    end


    %For inspection purposes (eg plotting), not speed
    % Returns mu = B\a and C = inv(B)
    function [mu,C] = get_mu_cov()
        mu = B\a;
        C = inv(B);
    end
    
    % This may be useful later. Returns new object, which is the
    % convolution of this Gaussian and another, represented by AE.
    % The calculation adds means and covariances and converts back to
    % natural parameters. For improved efficiency and accuracy, we use: 
    % inv( inv(B1)+inv(B2) ) = B1*inv(B1+B2)*B2 = B2*inv(B1+B2)*B1  
    function CE = convolve(AE)
        [a1,B1] = deal(a,B); %rename, just for clear code
        [a2,B2] = AE.getNatParams();
        chol12 = chol(B1+B2);
        solve = @(rhs) chol12\(chol12.'\rhs); % inv(B1+B2)*rhs
        newB = B1*solve(B2);  %this is inv(inv(B1)+inv(B2))  
        newa = B2*solve(a1) + B1*solve(a2); %newB * (B1\a1 + B2\a2 )
        CE = create_plain_metaEmb(newa,newB);
    end

    
end