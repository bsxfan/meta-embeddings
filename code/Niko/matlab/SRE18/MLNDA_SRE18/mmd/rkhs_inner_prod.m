function y = rkhs_inner_prod(sigma,B1,a1,B2,a2)

    dim = length(a1);
    I = eye(dim);
    Bconv = (I+sigma*B1)\B1;   % inv(sigmaI + inv(B1))
    aconv = (I+sigma*B1)\a1;    % Bconv*(B1\a)

    y = gauss_prod_int(aconv,Bconv,a2,B2);
    
end

function y = log_gauss_norm(B,a)
    [logd,mu,back] = logdet_solveLU(B,a);
    y = (logd-mu.'*a)/2;
    
end


function y = gauss_prod_int(a1,B1,a2,B2)

    y = exp(log_gauss_norm(B1,a1) + log_gauss_norm(B2,a2) ...
            - log_gauss_norm(B1+B2,a1+a2) );
    
  
end