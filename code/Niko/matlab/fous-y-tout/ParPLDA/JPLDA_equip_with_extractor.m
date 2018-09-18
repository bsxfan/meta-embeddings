function model = JPLDA_equip_with_extractor(model)
% Equip SPLDA model with function handles for runtime functionality

    V = model.V;
    L = model.L;
    mu = model.mu;
    W = model.W;
    
    VW = V.'*W;
    LW = L.'*W;
    A = VW*V;
    Atilde = LW*L;
    B = VW*L;
    
    D = Atilde + eye(size(Atilde));
    Dtilde = A + eye(size(A));
    
    BD = B\D;
    BDtilde = B.'/Dtilde;
    Pz = A - BD*B.';
    Pl = Atilde - BDtilde*B.';
    
    
    model.extractME = @extractME;
    
    
    
    function [mezl,mez,mel] = extractME(X)
        X0 = bsxfun(@minus,X,mu);
        a = VW*X0;
        b = LW*X0;
        mezl.P = [A B; B.' C];
        mezl.F = [a;b];
        
        mez.P = Pz;
        mez.F = a-BD*b;
        
        mel.P = Pl;
        mel.F = b-BDtilde*a;
        
        
    end




end