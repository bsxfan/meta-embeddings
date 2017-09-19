function [X,precisions] = sample_speaker(z,F,k,n)
% sample n observations of speaker with identity variable z

    if nargin==0
        test_this();
        return;
    end

    precisions = -mean(log(rand(k,n)),1);   %samples from Gamma(k,1/k): mean value at 1 and mode at (k-1)/k
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