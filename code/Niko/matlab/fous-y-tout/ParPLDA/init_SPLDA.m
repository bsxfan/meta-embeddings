function [model,X] = init_SPLDA(X,Labels,zdim)
% Ferrer's SPLDA initialization (arxiv.org/abs/1803.10554)
% Inputs:
%   X: m-by-n, training data: n i-vectors of dimension m
%   Labels: k-by-n, sparse, logical, speaker label matrix, with one-hot
%           columns. (There are k speakers.)
%   zdim: speaker space dimensionality
%
%   Outputs:
%     model
%     X: [optional] training data with mean removed


    mu = mean(X,2);
    X = bsxfun(@minus,X,mu);
    model.mu = mu;


    [k,n] = size(Labels);
    
    counts = sum(Labels,2);    % k-by-1
    assert(all(counts),'empty speaker not allowed'); 
    
    Means = bsxfun(@rdivide,X*Labels.',counts.');     % m-by-k speaker means

    Cb = (Means*Means.')/k;   % between-speaker cov
    
    Delta = X - Means*Labels;
    Cw = (Delta*Delta.')/n;   % within-speaker cov
    
    W = inv(Cw);              % within speaker precision
    model.W = W;

    [E,L] = eig(Cb);
    L = diag(L);
    [L,ii] = sort(L,'descend');
    L = L(1:zdim);
    E = E(:,ii(1:zdim));
    V = bsxfun(@times,E,sqrt(L).');
    model.V = V;
    



end