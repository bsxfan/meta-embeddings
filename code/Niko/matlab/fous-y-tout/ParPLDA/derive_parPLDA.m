function model = derive_parPLDA(modelXY,xdim,ydim)
% Inputs:
%    modelXY: SPLDA model for stacked [X;Y] data
%    xdim,ydim: dimensionality of X and Y

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
    model = SPLDA_equip_with_scoring(model,zdim);
    

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