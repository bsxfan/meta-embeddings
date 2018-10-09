function [y,back] = gauss_mmd(f,P,sigma)
% Computes MMD (maximum mean discrepancy), aka kernel divergence, between 
% a given multivariate normal distribution and the standard normal. The kernel is RBF, with given scale
% parameter.
%
%   mmd =   0.5 int int Q(x) K(x,y) Q(y) dx dy
%         + 0.5 int int S(x) K(x,y) S(y) dx dy  
%         -     int int S(x) K(x,y) Q(y) dx dy  
%  
% where 
%   Q(x) = ND(x | P\f, inv(P) ) 
% is the input Gaussian density and 
%   S(x) = N(x | 0, I). 
% The kernel is 
%   K(x,y) = ND(x | y, sigma I).
%
% Notice that by convolution of Gaussians:
%   int S(y) K(x,y) dy = ND(x | 0, (sigma+1) I), and
%   int Q(y) K(x,y) dy = ND(x | P\f, inv(P) + sigma I)

    dim = length(f);
    I = eye(dim);
    nulvec = zeros(dim,1);
    
    PI = inv(inv(P)+I);

    y = 0.5*gauss_prod_int(nulvec,I,nulvec,I/(sigma+1)) + ...
        0.5*gauss_prod_int(f,P,f,PI) - ...
        gauss_prod_int(f,P,nulvec,I/(sigma+1));
end

function y = log_gauss_norm(B,a)
    [logd,mu,back] = logdet_solveLU(B,a);
    y = (logd-mu.'*a)/2;
    
end


function y = gauss_prod_int(a1,B1,a2,B2)

    y = exp(log_gauss_norm(B1,a1) + log_gauss_norm(B2,a2) ...
            - log_gauss_norm(B1+B2,a1+a2) );
    
  
end