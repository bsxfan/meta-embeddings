function dP = create_diagonalized_precision(P)

    if nargin==0 
        test_this();
        return;
    end
        

    [V,E] = eig(P);
    E = diag(E);

    dP.logdet_I_plus_nP = @logdet_I_plus_nP;
    dP.solve_I_plus_nP = @solve_I_plus_nP;
    
    function y = logdet_I_plus_nP(n)
        nE = bsxfun(@times,n,E);
        y = sum(log1p(nE),1);
    end

    function X = solve_I_plus_nP(n,R)
        nE = bsxfun(@times,n,E);
        X = V*bsxfun(@ldivide,1+nE,V'*R);
    end
    

end

function test_this()

    P = randn(3,4);
    P = P*P.';
    I = eye(size(P));
    
    dP = create_diagonalized_precision(P);
    n = randi(5,1,2);
    
    M1 = I + n(1)*P;
    M2 = I + n(2)*P;
    R = randn(3,2);
    [log(det(M1)),log(det(M2)),dP.logdet_I_plus_nP(n)]
    [M1\R(:,1),M2\R(:,2),dP.solve_I_plus_nP(n,R)]
    

end