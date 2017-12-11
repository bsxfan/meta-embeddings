function [X,precisions] = sample_HTnoise(nu,dim,n)
% Sample n heavy-tailed observations of speaker with identity variable z.
% Inputs:
%   nu: integer nu >=1, degrees of freedom of resulting t-distribution
%   n: number of samples
%
% Output:
%   X: dim-by-n samples
%   precisions: 1-by-n, the hidden precisions

    if nargin==0
        test_this();
        return;
    end
    
    precisions = mean(randn(nu,n).^2,1);  
    std = 1./sqrt(precisions);
    
    X = bsxfun(@times,std,randn(dim,n));




end

function test_this()

  close all;
  
  [X,precisions] = sample_HTnoise(2,2,1000);
  
  figure;
  plot(X(1,:),X(2,:),'.');

  figure;
  plot(sum(X.^2,1),1./precisions,'.');
  
  
  
end