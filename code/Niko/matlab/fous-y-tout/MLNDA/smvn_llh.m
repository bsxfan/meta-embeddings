function [y,back] = smvn_llh(R)

    if nargin==0
        test_this();
        return;
    end

    y = (-0.5)*R(:).'*R(:);
    back = @(dy) (-dy)*R;

end

function test_this()

    R = randn(3,5);
    testBackprop(@smvn_llh,{R});

end