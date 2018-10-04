function [M,back] = diag2full(d)

    if nargin==0
        test_this();
        return;
    end

    dim = length(d);
    
    M = sparse(1:dim,1:dim,d,dim,dim);
    back = @back_this;
    
    function [dd] = back_this(dM)
        dd = diag(dM);
    end


end

function test_this()

    d = randn(5,1);
    testBackprop(@diag2full,{d});
    
    
end