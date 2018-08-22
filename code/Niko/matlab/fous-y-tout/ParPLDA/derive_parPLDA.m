function model = derive_parPLDA(modelXY,xdim,ydim)
% Inputs:
%    modelXY: SPLDA model for stacked [X;Y] data
%    xdim,ydim: dimensionality of X and Y
%
% Outputs: 
%     model: a struct, with function handles for:
%       - Extracting mutually compatible meta-embeddings from any of X, Y or
%       [X;Y].
%       - Scoring any such meta-embedding against any other.
%     
%     Meta-embeddings are extracted with any of the three combinations:
%       me = model.extractME(X,Y)
%       me = model.extractME(X,[])
%       me = model.extractME([],Y)
%     where:
%         X: m-by-n1 and Y: d-by-n1, for example i-vectors and x-vectors. If
%                                    both are given, we need the same number
%                                    of vectors in both X and Y sets.
%         me: struct representing Gaussian meta-embeddings, n1 of them, where:
%         me.P: zdim-by-zdim positive semi-definite precision, common to all 
%               the meta-embeddings in this struct.
%         me.F: zdim-by-n1, first-order natural parameters for each the n1 
%               meta-embeddings '
%
%      Meta-embeddings from the same speaker can be pooled, to form 
%      `speaker models', when multiple enrollment sessions are available.
%      For example, for each of n2 speakers, let X1: m-by-n2 
%      represent the first enrollment sessions, in form of i-vectors. Also
%      let X2: m-by-n2 and Y2: d-by-n2, represent the second enrollment sessions 
%      for each speaker, but available in both forms. The enrollment
%      meta-embeddings for these speakers can be formed as:
%
%        enrollments = model.poolME(model.extractME(X1,[]),model.extractME(X2,Y2));
% 
%      Note pooling is appropriate when data is independent, given the speaker. If
%      X1 and Y1 are extracted from the same set of recordings, they are not
%      independent. In this case: me = model.extractME(X1,Y1) is correct,
%      while me = model.poolME(model.extractME(X1,[]),model.extractME([],Y1)) is
%      incorrect. Conversely, if X1 and Y2 are extracted from different
%      sets of recordings (of the same speakers), then pooling is correct: 
%        me = model.poolME(model.extractME(X1,[]),model.extractME([],Y2)).
%
%
%      For scoring, there are two functions for effiently scoring sets of trials:
%        - model.scoreTrails(enroll,test): scores n enrollment meta-embeddings
%                             against the corresponding n test meta-embeddings.
%                             This returns a vector of n scores.
%        - model.scoreMatrix(enroll,test): scores m enrollment meta-embeddings against
%                             **each** of n test meta-embeddings. This
%                             returns an m-by-n matrix of scores.
%      Note the argument names 'enroll' and 'test' are arbitrary. Scoring
%      is symmetric. These arguments are meta-embedding structs, extracted
%      (and optionally pooled) as described above.

    modelXY = SPLDA_equip_with_extractor(modelXY);
    
    V = modelXY.V;
    W = modelXY.W;
    mu = modelXY.mu;
    zdim = size(V,2);
    
    xx = 1:xdim;
    yy = xdim+(1:ydim);
    
    R = W(xx,xx);
    S = W(yy,yy);
    C = W(xx,yy);
    
    modelX.V = V(xx,:);
    modelX.W = R - C*(S\C.');
    modelX.mu = mu(xx);
    modelX = SPLDA_equip_with_extractor(modelX);
    
    modelY.V = V(yy,:);
    modelY.W = S - C.'*(R\C);
    modelY.mu = mu(yy);
    modelY = SPLDA_equip_with_extractor(modelY);

    model.extractME = @extractME;
    model = equip_with_GME_scoring(model,zdim);
    

    function me = extractME(X,Y) 
        if ~isempty(X) && ~isempty(Y)
            me = modelXY.extractME([X;Y]);
        elseif ~isempty(X) && isempty(Y)
            me = modelX.extractME(X);
        elseif isempty(X) && ~isempty(Y)
            me = modelY.extractME(Y);
        else % no data
            me.P = 0;
            me.F = 0;
        end
    end


end