function [SGMEP,meand] = scaled_GME_precision(B)

    if nargin==0
        test_this();
        return;
    end

    dim = size(B,1);

    [V,D] = eig(B);  % B = VDV'
    d = diag(D);
    meand = mean(d);
    %D = sparse(D);
    %I = speye(dim);
    
    SGMEP.logdet = @logdet;
    SGMEP.solve = @solve;
    
    function [y,back] = logdet(beta)
        betad = beta*d;
        y = sum(log1p(betad),1);
        back = @(dy) dy*sum(d./(1+betad),1);
    end


    function [Y,back] = solve(RHS,beta)
        betad = beta*d;
        Y = V*bsxfun(@ldivide,betad+1,V.'*RHS);
        back = @(dY) back_solve(dY,Y,beta); 
    end

    function [dRHS,dbeta] = back_solve(dY,Y,beta)
        dRHS = solve(dY,beta);
        if nargout >= 2
          dA = (-dRHS)*Y.';
          dbeta = sum(sum(dA));
          %dbeta = -(sum(dRHS,1))*sum(Y,1).';
        end
    end



end

function test_this()

    close all;

    fprintf('Test function values:\n');
    dim = 5;
    RHS = rand(dim,1);
    
    %R = randn(dim,floor(1.1*dim));B = R*R.';B = B/trace(B);
    R = randn(dim,dim);B = R*R.';B = B/trace(B);
    I = eye(dim);
    
    [SGMEP,meand] = scaled_GME_precision(B);
    
    beta = rand/rand;
    [log(det(I+beta*B)),SGMEP.logdet(beta)]
    
    [(I+beta*B)\RHS,SGMEP.solve(RHS,beta)]

    doplot = false;
    if doplot
        beta = 0.01:0.01:200;
        y = zeros(size(beta));
        for i=1:length(beta)
            y(i) = SGMEP.logdet(beta(i));
        end
        1/meand
        plot(log(1/meand+beta),y);
    end
    
    
    fprintf('\n\n\nTest logdet backprop (complex step) :\n');
    testBackprop(SGMEP.logdet,beta);    

    fprintf('\n\n\nTest logdet backprop (real step) :\n');
    testBackprop_rs(SGMEP.logdet,beta,1e-4);    

    fprintf('\n\n\nTest solve backprop (complex step) :\n');
    testBackprop(SGMEP.solve,{RHS,beta},{1,0});    
    
    fprintf('\n\n\nTest solve backprop (real step) :\n');
    testBackprop_rs(SGMEP.solve,{RHS,beta},1e-4,{1,0});    
    
end


