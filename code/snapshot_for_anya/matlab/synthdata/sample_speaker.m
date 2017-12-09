function [X,precisions] = sample_speaker(z,F,k,n,chi_sq)
% Sample n heavy-tailed observations of speaker with identity variable z.
% Inputs:
%   z: d-by-1 speaker identity variable
%   F: D-by-d factor loading matrix
%   k: integer, k>=1, where nu=2k is degrees of freedom of resulting
%      t-distribution
%   n: number of samples
%   chi_sq: [optional] If given and true, then precisions are sampled from
%           chi^2 with DF: nu = k*2. In this case, k*2 must be an integer,
%           so for example k=0.5 is valid and gives Cauchy samples. 
%
% Output:
%   X: D-by-n samples
%   precisions: 1-by-n, the hidden precisions

    if nargin==0
        test_this();
        return;
    end
    
    if ~exist('n','var') || isempty(n)
        n = size(z,2);
    end
    
    if exist('chi_sq','var') && ~isempty(chi_sq) && chi_sq
        % sample Chi^2, with DF = nu=2k, scaled by 1/nu, so that mean = 1. 
        nu = 2*k;
        precisions = mean(randn(nu,n).^2,1);  
    else %Gamma
        % Sample n precisions independently from Gamma(k,k), which has mean = 1
        % mode = (k-1)/k.
        precisions = -mean(log(rand(k,n)),1);   
    end
    
    std = 1./sqrt(precisions);
    
    dim = size(F,1);
    Y = bsxfun(@times,std,randn(dim,n));
    X = bsxfun(@plus,F*z,Y);




end

function test_this()

  close all;
  
  z = 0;
  F = zeros(100,1);
  k = 5;
  [X,precisions] = sample_speaker(z,F,k,1000);
  
  figure;
  plot(X(1,:),X(2,:),'.');

  figure;
  plot(sum(X.^2,1),1./precisions,'.');
  
  
  
end