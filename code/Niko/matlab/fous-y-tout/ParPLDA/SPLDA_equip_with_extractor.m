function model = SPLDA_equip_with_extractor(model)
% Equip SPLDA model with function handles for runtime functionality

    V = model.V;
    mu = model.mu;
    W = moddel.W;
    
    VW = V.'*W;
    VWV = VW*V;
    
    zdim = size(V,2);
    I = speye(zdim);
    
    
    model.extractME = @extractME;
    model.poolME = @poolME;
    model.logExpectation = @logExpectation;
    model.scoreMatrix = @scoreMatrix;
    model.scoreTrials = @scoreTrials;
    
    
    
    function me = extractME(X)
        me.P = VWV;
        me.F = VW*bsxfun(@minus,X,mu);
    end




end