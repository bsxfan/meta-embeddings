function [y,back] = rkhs_normsq_slow(sigma,A,b,B)
% RKHS inner products of multivariate Gaussians onto standard normal. The
% RKHS kernel is K(x,y) = ND(x|y,sigma I). The first order natural
% parameters of the Gaussians are in the columns of A. The precisions are
% proportional to the fixed B, with scaling contants in the vector b.


    [dim,n] = size(A);
    
    I = eye(dim);
    
    y = zeros(1,n);

    for i=1:n
        a = A(:,i);
        Bi = b(i)*B;
        
        Bconv = (I+sigma*Bi)\Bi;   % inv(sigmaI + inv(B))
        aconv = (I+sigma*Bi)\a;    % Bconv*(B\a)
        
        lgn1 = log_gauss_norm(Bconv,aconv);
        lgn2 = log_gauss_norm(Bi,a);
        lgn12 = log_gauss_norm(Bconv + Bi,a+aconv);
        y(i) = exp(lgn1 + lgn2 - lgn12);
    end


end