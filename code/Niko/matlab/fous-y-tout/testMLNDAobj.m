function [y,back] = testMLNDAobj(T,labels,F,W,fi,params)

    [R,logdetJ,back2] = fi(params,T);
    [llh,back1] = splda_llh(R,labels,F,W);
    y = logdetJ - llh;
    
    back = @back_this;
    
    function dparams = back_this(dy)
        dlogdetJ = dy;
        dR = back1(-dy);
        dparams = back2(dR,dlogdetJ);
    end


end