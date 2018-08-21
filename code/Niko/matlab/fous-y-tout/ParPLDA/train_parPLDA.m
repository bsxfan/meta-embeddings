function model = train_parPLDA(X,Y,Labels,zdim)
% Inputs:
%   X: m-by-n, training data: n i-vectors of dimension m
%   X: d-by-n, training data: n ?-vectors of dimension d
%   Labels: k-by-n, sparse, logical, speaker label matrix, with one-hot
%           columns. (There are k speakers.)
%   zdim: speaker space dimensionality
%
%   Output:
%     model

    [xdim,nx] = size(X);
    [ydim,ny] = size(Y);
    [~,n] = size(Labels);
    assert(nx==ny && ny == n,'illegal arguments');
    

    modelXY = init_SPLDA([X;Y],Labels,zdim);
    
    model = derive_parPLDA(modelXY,xdim,ydim);
    

end