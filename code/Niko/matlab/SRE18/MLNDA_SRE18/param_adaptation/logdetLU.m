function [y,back] = logdetLU(M)

    if nargin==0
        test_this();
        return;
    end

    [L,U] = lu(M);
    y = sum(log(diag(U).^2))/2;
    
    back = @back_this;
    
    function dM = back_this(dy)
        %dM = dy*(inv(U)/L).';    
        dM = dy*(L.'\inv(U.'));    
    end


end

function test_this()

    dim = 5;
    M = randn(dim);
    
    testBackprop(@logdetLU,M);

end