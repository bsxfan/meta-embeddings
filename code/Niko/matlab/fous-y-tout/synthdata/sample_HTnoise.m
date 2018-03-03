function [X,precisions] = sample_HTnoise(nu,dim,n,W)
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
    
    
    if ~exist('W','var') || isempty(W)
        cholW = speye(dim);
    else
        cholW = chol(W);    
    end
    
    
    
    precisions = mean(randn(nu,n).^2,1);  
    std = 1./sqrt(precisions);
    
    X = cholW*bsxfun(@times,std,randn(dim,n));




end

function test_this()

  close all;
  
  dim = 2;
  nu = 2;
  W = randn(2,3); W = W*W.';
  
  [X,precisions] = sample_HTnoise(nu,dim,1000,W);
  
  figure;
  plot(X(1,:),X(2,:),'.');axis('equal');axis('square');

  %figure;
  %plot(sum(X.^2,1),1./precisions,'.');
  
  
  
end