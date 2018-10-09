function [y,back] = MLNDAobj(T,labels,F,W,fi,params,nu)

    ht = exist('nu','var') && ~isempty(nu) && ~isinf(nu);

    [R,logdetJ,back2] = fi(params,T);
    
    
    if ht
        [llh,back1] = htplda_llh(R,labels,F,W,nu);
    else
        [llh,back1] = splda_llh(R,labels,F,W);
    end
    
    
    y = logdetJ - llh;
    
    back = @back_this;
    
    function dparams = back_this(dy)
        dlogdetJ = dy;
        dR = back1(-dy);
        dparams = back2(dR,dlogdetJ);
    end


end