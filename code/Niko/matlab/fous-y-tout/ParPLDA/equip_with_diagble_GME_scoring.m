function model = equip_with_diagble_GME_scoring(model)
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



    if nargin==0
        test_this();
        return;
    end

    
    model.poolDME = @poolDME;
    model.logExpectationD = @logExpectationD;
    model.scoreMatrixD = @scoreMatrixD;
    model.scoreTrialsD = @scoreTrialsD;
    model.estimateZD = @estimateZD;
    

    function me = poolDME(me1,me2)
        %assert(isequal(me1.P,me2.P));  how to test for handle equality
        me.n = me1.n + me2.n;
        me.F = me1.F + me2.F;
        me.dP = me1.dP;
    end

    function y = logExpectationD(me)
        dP = me.dP;
        logdet = dP.logdet_I_plus_nP(me.n);
        Z = estimateZD(me);
        y = ( sum(Z.*me.F,1) - logdet )/2;
    end
     
    function Z = estimateZD(me)
        dP = me.dP;
        Z = dP.solve_I_plus_nP(me.n,me.F);
    end

    function LLR = scoreMatrixD(left,right)
        dP = left.dP;
        sleft = logExpectationD(left);
        sright = logExpectationD(right);
        
        m = length(sleft);
        n = length(sright);
        LLR = zeros(m,n);
        for i=1:m
            ni = left.n(i);
            Fi = left.F(:,i);
            nn = ni + right.n;
            FF = bsxfun(@plus,Fi,right.F);
            logdet = dP.logdet_I_plus_nP(nn);
            ZZ = dP.solve_I_plus_nP(nn,FF);
            LLR(i,:) = ( sum(ZZ.*FF,1) - logdet )/2 - sleft(i) - sright;
        end
        
        
        
    end


    function llr = scoreTrialsD(enroll,test)
        llr = logExpectationD(poolDME(enroll,test)) - logExpectationD(enroll) - logExpectationD(test);
    end




end


function test_this

    dim = 10;
    zdim = 2;
    mu = randn(dim,1);
    V = randn(dim,zdim);
    W = randn(dim,dim+1); W = W * W.';
    model.mu = mu;
    model.V = V;
    model.W = W;
    
    model = SPLDA_equip_with_extractor(model);
    model = SPLDA_equip_with_diagble_extractor(model);
    model = equip_with_GME_scoring(model,zdim);
    model = equip_with_diagble_GME_scoring(model);
    
    m = 3;
    %n = m; 
    n = 4;
    enroll1 = randn(dim,m);
    enroll2 = randn(dim,m);
    test = randn(dim,n);
    
    E1 = model.extractME(enroll1);
    E2 = model.extractME(enroll2);
    E = model.poolME(E1,E2);
    T = model.extractME(test);
    
    %model.scoreTrials(E1,T)
    %model.logExpectation(E)
    model.scoreMatrix(E,T)
    
    E1 = model.extractDME(enroll1);
    E2 = model.extractDME(enroll2);
    E = model.poolDME(E1,E2);
    T = model.extractDME(test);
    
    %model.scoreTrialsD(E1,T)
    %model.logExpectationD(E)
    model.scoreMatrixD(E,T)
    
    
    
    
    
    
    

end

