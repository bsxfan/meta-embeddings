function [logdet,back] = logdetNice2(sigma,R,D)

    if nargin==0
        test_this();
        return;
    end

    RR = R.'*R;
    [Ld,Ud] = lu(D);
    invD = inv(Ud)/Ld;
    S = RR/sigma + invD;
    [L,U] = lu(S);
    dim = size(R,1);
    logdet = ( sum(log(diag(U).^2)) + sum(log(diag(Ud).^2)) + dim*log(sigma^2) ) /2;
    
    back = @back_this;
    
    function [dsigma,dR,dD] = back_this(dlogdet)
        dS = dlogdet*(inv(U)/L).';
        dD = dlogdet*invD.';
        dsigma = dim*dlogdet/sigma;
        dR = R*(dS + dS.')/sigma;
        dsigma = dsigma - (RR(:).'*dS(:))/sigma^2;
        dinvD = dS;
        dD = dD - D.'\(dinvD/D.');
    end

end

function test_this()

    dim = 5;
    rank = 2;
    
    sigma = randn;
    R = randn(dim,rank);
    D = randn(rank);
    
    M = sigma*eye(dim) + R*D*R.';
    
    [log(abs(det(M))),logdetNice2(sigma,R,D)]
    
    
    testBackprop(@logdetNice2,{sigma,R,D})
    

end