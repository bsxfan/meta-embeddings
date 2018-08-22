function model = SPLDA_equip_with_extractor(model)
% Equip SPLDA model with function handles for runtime functionality

    V = model.V;
    mu = model.mu;
    W = model.W;
    
    VW = V.'*W;
    VWV = VW*V;
    
    
    model.extractME = @extractME;
    
    
    
    function me = extractME(X)
        me.P = VWV;
        me.F = VW*bsxfun(@minus,X,mu);
    end




end