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
%   sdim: speaker space dimensionality
%   cdim: condition space dimensionality


    [spldaC,X] = init_SPLDA(X,Cond,cdim);  %mean is removed from X
    model.mu = splda.mu;
    W = spldaC.V;                          %condition factor loading matrix
    model.W = W;
    
    spldaC.mu = 0;
    spldaC = SPLDA_equip_with_diagble_extractor(spldaC);
    spldaC = equip_with_diagble_GME_scoring(spldaC);
    me = spldaC.extractDME(X);
    
    %pool
    me.n = me.n*Cond.';              %1 by m
    me.F = me.F*Cond.';              %cdim-by-m
    
    C = spldaC.estimateZD(me);       %cdim-by-m
    
    %compensate for condition
    X = X - W*C*Cond;        
    
    spldaS = init_SPLDA(X,Spkrs,sdim);
    model.V = spldaS.V;
    model.D
      

end