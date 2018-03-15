function CA = asEig_svd(A)

    if nargin==0
        test_this();
        return;
    end

    if isreal(A)
        [V,D] = eig(A);   %V*D*V' = A
        D = diag(D);
        r = true;
    else
        [U,S,V] = svd(A); % U*S*V' = A
        S = diag(S);
        r = false;
    end
    
    dim = size(A,1);
    
    CA.logdet = @logdet;
    CA.solve = @solve;
    
    
    function [y,back] = logdet()
        if r
            y = sum(log(D));
        else
            y = sum(log(S));
        end
        back = @(dy) solve(dy*speye(dim));
    end




    function [Y,back] = solve(RHS)
        if r
            Y = V*bsxfun(@ldivide,D,V.'*RHS);
        else
            Y = V*bsxfun(@ldivide,S,U.'*RHS);
        end
        back = @(dY) back_solve(dY,Y);
    end

    function Y = solveT(RHS)  %A'\RHS, for LU case
        Y = U*bsxfun(@ldivide,S,V.'*RHS);
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
    CA = asEig_svd(A*A.');
    [y,back1] = CA.logdet();
    sym = @(DY) DY + DY.';
    back =@(dy) sym(back1(dy))*A;
end

function [Y,back] = solvetestfun(RHS,A)
    CA = asEig_svd(A*A.');
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
    
    
    CA = asEig_svd(A);
    
    [log(det(A)),CA.logdet()]
    [A\RHS,CA.solve(RHS)]


    A = complex(randn(dim),zeros(dim));
    CA = asEig_svd(A);
    
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


