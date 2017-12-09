function CA = asChol(A)

    if nargin==0
        test_this();
        return;
    end

    if isreal(A)
        C = chol(A);   %C'C = A
        r = true;
    else
        [L,U] = lu(A); % LU = A
        r = false;
    end
    
    dim = size(A,1);
    
    CA.logdet = @logdet;
    CA.solve = @solve;
    
    
    function [y,back] = logdet()
        if r
            y = 2*sum(log(diag(C)));
        else
            y = sum(log(diag(U).^2))/2;
        end
        back = @(dy) solve(dy*speye(dim));
    end




    function [Y,back] = solve(RHS)
        if r
            Y = C\(C'\RHS);
        else
            Y = U\(L\RHS);
        end
        back = @(dY) back_solve(dY,Y);
    end

    function Y = solveT(RHS)  %A'\RHS, for LU case
        Y = L.'\(U.'\RHS);
    end

    function [dRHS,dA] = back_solve(dY,Y)
        if r
            dRHS = solve(dY); 
            if nargout >= 2
                dA = (-dRHS)*Y.';
            end
        else 
            dRHS = solveT(dY);
            if nargout >= 2
                dA = (-dRHS)*Y.';
            end
        end
    end




end


function [y,back] = logdettestfun(A)
    CA = asChol(A*A.');
    [y,back1] = CA.logdet();
    sym = @(DY) DY + DY.';
    back =@(dy) sym(back1(dy))*A;
end

function [Y,back] = solvetestfun(RHS,A)
    CA = asChol(A*A.');
    [Y,back1] = CA.solve(RHS);
    
    back =@(dY) back_solvetestfun(dY);
    
    function [dRHS,dA] = back_solvetestfun(dY)
        [dRHS,dAA] = back1(dY);
        dA = (dAA+dAA.')*A;
    end
end


function test_this()

    fprintf('Test function values:\n');
    dim = 5;
    RHS = rand(dim,1);
    
    A = randn(dim);A = A*A';
    
    
    CA = asChol(A);
    
    [log(det(A)),CA.logdet()]
    [A\RHS,CA.solve(RHS)]


    A = complex(randn(dim),zeros(dim));
    CA = asChol(A);
    
    [log(abs(det(A))),CA.logdet()]
    [A\RHS,CA.solve(RHS)]

    
    
    A = randn(dim,2*dim);A = A*A';
    fprintf('\n\n\nTest logdet backprop (complex step) :\n');
    testBackprop(@logdettestfun,A);    
    fprintf('\n\n\nTest logdet backprop (real step) :\n');
    testBackprop_rs(@logdettestfun,A,1e-4);    
    
    
    fprintf('\n\n\nTest solve backprop (complex step) :\n');
    testBackprop(@solvetestfun,{RHS,A},{1,1});

    fprintf('\n\n\nTest solve backprop (real step) :\n');
    testBackprop_rs(@solvetestfun,{RHS,A},1e-4,{1,1});
    
end


