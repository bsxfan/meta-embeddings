function model = SPLDA_equip_with_diagble_extractor(model)
% Equip SPLDA model with a function handles for extracting diagonalizable
% meta-embeddings

    V = model.V;
    mu = model.mu;
    W = model.W;
    
    VW = V.'*W;
    dP = create_diagonalized_precision(VW*V);
    
    
    model.extractDME = @extractDME;
    
    
    
    function me = extractDME(X)
        me.dP = dP;
        me.F = VW*bsxfun(@minus,X,mu);
        me.n = ones(1,size(X,2));
    end




end