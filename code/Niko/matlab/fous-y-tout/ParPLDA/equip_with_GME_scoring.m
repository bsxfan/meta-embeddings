function model = equip_with_GME_scoring(model,zdim)
% Equip any model with function handles for runtime scoring
% functionality for Gaussian meta-embeddings (GMEs).
%
%   Inputs:
%     model: any struct. The struct members are not referenced in this code.
%            A number of method handles (described below) are added to the 
%            struct on output.
%     zdim: the speaker space dimensionality 
%
%
%
%     Meta-embeddings from the same speaker can be pooled, to form 
%     `speaker models', when multiple enrollment sessions are available.
%     For example, for each of n2 speakers, let X1: m-by-n2 
%     represent the first enrollment sessions, in form of i-vectors. Also
%     let X2: m-by-n2 and Y2: d-by-n2, represent the second enrollment sessions 
%     for each speaker, but available in both forms. The enrollment
%     meta-embeddings for these speakers can be formed as:
%
%       enrollments = model.poolME(model.extractME(X1,[]),model.extractME(X2,Y2));
%
%     Note pooling is appropriate when data is independent, given the speaker. If
%     X1 and Y1 are extracted from the same set of recordings, they are not
%     independent. In this case: me = model.extractME(X1,Y1) is correct,
%     while me = model.poolME(model.extractME(X1,[]),model.extractME([],Y1)) is
%     incorrect. Conversely, if X1 and Y2 are extracted from different
%     sets of recordings (of the same speakers), then pooling is correct: 
%       me = model.poolME(model.extractME(X1,[]),model.extractME([],Y2)).
%
%
%     For scoring, there are two functions for effiently scoring sets of trials:
%       - model.scoreTrails(enroll,test): scores n enrollment meta-embeddings
%                            against the corresponding n test meta-embeddings.
%                            This returns a vector of n scores.
%       - model.scoreMatrix(enroll,test): scores m enrollment meta-embeddings against
%                            **each** of n test meta-embeddings. This
%                            returns an m-by-n matrix of scores.
%     Note the argument names 'enroll' and 'test' are arbitrary. Scoring
%     is symmetric. These arguments are meta-embedding structs, extracted
%     (and optionally pooled) as described above.

    I = speye(zdim);
    
    model.poolME = @poolME;
    model.logExpectation = @logExpectation;
    model.scoreMatrix = @scoreMatrix;
    model.scoreTrials = @scoreTrials;
    model.estimateZ = @estimateZ;
    

    function me = poolME(me1,me2)
        me.P = me1.P + me2.P;
        me.F = me1.F + me2.F;
    end

    function y = logExpectation(me)
        cholIP = chol(I+me.P);
        logdet = 2*sum(log(diag(cholIP)));
        Z = cholIP\(cholIP'\me.F);
        y = ( sum(Z.*me.F,1) - logdet )/2;
    end
     
    function Z = estimateZ(me)
        cholIP = chol(I+me.P);
        Z = cholIP\(cholIP'\me.F);
    end

    function LLR = scoreMatrix(left,right)
        sleft = logExpectation(left);
        sright = logExpectation(right);
        
        P = left.P + right.P;
        cholIP = chol(I+P);
        logdet = 2*sum(log(diag(cholIP)));
        
        Zleft = cholIP\(cholIP'\left.F);
        Zright = cholIP\(cholIP'\right.F);
        
        LLR = left.F.'*Zright;
        LLR = bsxfun(@plus,LLR,(sum(left.F.*Zleft,1).' - logdet)/2 - sleft.');
        LLR = bsxfun(@plus,LLR,sum(right.F.*Zright,1)/2 - sright);
        
        
    end


    function llr = scoreTrials(enroll,test)
        llr = logExpectation(poolME(enroll,test)) - logExpectation(enroll) - logExpectation(test);
    end




end