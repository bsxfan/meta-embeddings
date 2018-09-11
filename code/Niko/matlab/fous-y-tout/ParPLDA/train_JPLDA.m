function model = train_JPLDA(X,Spkrs,Cond,sdim,cdim)
% Ferrer's JPLDA initialization (arxiv.org/abs/1803.10554)
%
% Inputs:
%   X: dim-by-n, training data, n i-vectors of dimension dim
%
%   Spkrs: k-by-n, sparse logical speaker labels, with one-hot columns for
%          k speakers.
%
%   Cond: m-by-n, sparse logical condition labels, with one-hot columns for
%         m conditions.
%

    [spldaC,X] = init_SPLDA(X,Cond,cdim);  %mean is removed from X
    model.mu = splda.mu;
    W = spldaC.V;                          %condition factor loading matrix
    model.W = W;
    
    spldaC = equip_with_GME_scoring(spldaC,cdim);
    C = spldaC.estimateZ(spldaC.extractME(X));       %cdim-by-n
    X = X - W*C; 

end