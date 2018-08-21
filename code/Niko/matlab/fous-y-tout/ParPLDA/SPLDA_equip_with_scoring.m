function model = SPLDA_equip_with_scoring(model,zdim)
% Equip SPLDA model with function handles for runtime functionality

    I = speye(zdim);
    
    model.poolME = @poolME;
    model.logExpectation = @logExpectation;
    model.scoreMatrix = @scoreMatrix;
    model.scoreTrials = @scoreTrials;
    

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
     
    function LLR = scoreMatrix(left,right)
        sleft = logExpectation(left);
        sright = logExpectation(right);
        
        P = left.P + right.P;
        cholIP = chol(I+me.P);
        logdet = 2*sum(log(diag(cholIP)));
        
        Zleft = cholIP\(cholIP'\left.F);
        Zright = cholIP\(cholIP'\right.F);
        
        LLR = left.F.'*zRight;
        LLR = bsxfun(@plus,LLR,(sum(left.F.*Zleft,1).' - logdet)/2 - sleft.');
        LLR = bsxfun(@plus,LLR,sum(right.F.*Zright,1).'/2 - sright);
        
        
    end


    function llr = scoreTrials(enroll,test)
        llr = logExpectation(poolME(enroll,test)) - logExpectation(enroll) - logExpectation(test);
    end




end