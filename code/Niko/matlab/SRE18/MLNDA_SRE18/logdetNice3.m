function [logdet,back] = logdetNice3(sigma,L,R)

    if nargin==0
        test_this();
        return;
    end

    [dim,rank] = size(L);
    
    RL = R.'*L;
    S = RL/sigma + eye(rank);
    [Ls,Us] = lu(S);
    logdet = ( sum(log(diag(Us).^2)) + dim*log(sigma^2) ) /2;
    
    back = @back_this;
    
    function [dsigma,dL,dR] = back_this(dlogdet)
        
        %logdet = ( sum(log(diag(Us).^2)) + dim*log(sigma^2) ) /2;
        dS = dlogdet*(inv(Us)/Ls).';
        dsigma = dim*dlogdet/sigma;
        
        %S = RL/sigma + eye(rank)
        dRL = dS/sigma;
        dsigma = dsigma - (RL(:).'*dS(:))/sigma^2;
        
        %RL = R.'*L;
        dL = R*dRL;
        dR = L*dRL.';
    end

end

function test_this()

    dim = 5;
    rank = 2;
    
    sigma = randn;
    R = randn(dim,rank);
    L = randn(dim,rank);
    
    M = sigma*eye(dim) + L*R.';
    
    [log(abs(det(M))),logdetNice3(sigma,L,R)]
    
    
    testBackprop(@logdetNice3,{sigma,L,R})
    

end