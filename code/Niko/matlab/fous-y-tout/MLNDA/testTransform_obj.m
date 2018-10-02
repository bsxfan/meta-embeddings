function [y,back] = testTransform_obj(T,fi,params)

    if nargin==0
        test_this();
        return;
    end

    [R,logdetJ,back2] = fi(params,T);
    [llh,back1] = smvn_llh(R);
    y = logdetJ - llh;
    
    back = @back_this;
    
    function dparams = back_this(dy)
        dlogdetJ = dy;
        dR = back1(-dy);
        dparams = back2(dR,dlogdetJ);
    end


end

function test_this()

    R = randn(3,100);

    [f,fi] = create_scalTrans();
    scal = pi;
    T = f(scal,R);
    
    ss = scal*exp(-1:0.01:5);
    y = zeros(size(ss));
    for i=1:length(ss)
        y(i) = testTransform_obj(T,fi,ss(i));
    end
    close all;
    semilogx(ss,y);hold;
    semilogx(scal,testTransform_obj(T,fi,scal),'*r');



end