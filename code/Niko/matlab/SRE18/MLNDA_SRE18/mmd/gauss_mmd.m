function [y,back] = gauss_mmd(a,B,sigma)
% Computes MMD (maximum mean discrepancy), aka kernel divergence, between 
% a given multivariate normal distribution and the standard normal. The kernel is RBF, with given scale
% parameter.
%
%   mmd =   0.5 int int Q(x) K(x,y) Q(y) dx dy
%         + 0.5 int int S(x) K(x,y) S(y) dx dy  
%         -     int int S(x) K(x,y) Q(y) dx dy  
%  
% where 
%   Q(x) = ND(x | B\a, inv(B) ) 
% is the input Gaussian density and 
%   S(x) = N(x | 0, I). 
% The kernel is 
%   K(x,y) = ND(x | y, sigma I).
%
% Notice that by convolution of Gaussians:
%   int S(y) K(x,y) dy = ND(x | 0, (sigma+1) I), and
%   int Q(y) K(x,y) dy = ND(x | B\a, inv(B) + sigma I)

    dim = length(a);
    I = eye(dim);
    nul = zeros(dim,1);

    y = rkhs_inner_prod(sigma,I,nul,I,nul) + rkhs_inner_prod(sigma,B,a,B,a) ...
        - 2*rkhs_inner_prod(sigma,B,a,I,nul);


end