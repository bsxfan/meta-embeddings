function [logdet,back] = logdetNice(sigma,R,d)

    if nargin==0
        test_this();
        return;
    end

    RR = R.'*R;
    S = RR/sigma + diag(1./d);
    [L,U] = lu(S);
    dim = size(R,1);
    logdet = ( sum(log(diag(U).^2)) + sum(log(d.^2)) + dim*log(sigma^2) ) /2;
    
    back = @back_this;
    
    function [dsigma,dR,dd] = back_this(dlogdet)
        dS = dlogdet*(inv(U)/L).';
        dd = dlogdet./d;
        dsigma = dim*dlogdet/sigma;
        dR = R*(dS + dS.')/sigma;
        dsigma = dsigma - (RR(:).'*dS(:))/sigma^2;
        dd = dd - diag(dS)./d.^2;
    end

end

function test_this()

    dim = 5;
    rank = 2;
    
    sigma = randn;
    R = randn(dim,rank);
    d = randn(rank,1);
    
    M = sigma*eye(dim) + R*diag(d)*R.';
    
    [log(abs(det(M))),logdetNice(sigma,R,d)]
    
    
    testBackprop(@logdetNice,{sigma,R,d})
    

end