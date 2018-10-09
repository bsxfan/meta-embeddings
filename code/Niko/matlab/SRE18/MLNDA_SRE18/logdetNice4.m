function [logdet,back] = logdetNice4(D,L,R)

    if nargin==0
        test_this();
        return;
    end

    [~,rank] = size(L);
    
    DL = bsxfun(@ldivide,D,L);
    RDL = R.'*DL;
    S = RDL + eye(rank);
    [Ls,Us] = lu(S);
    logdet = ( sum(log(diag(Us).^2)) + sum(log(D.^2)) ) /2;
    
    back = @back_this;
    
    function [dD,dL,dR] = back_this(dlogdet)
        
        %logdet = ( sum(log(diag(Us).^2)) + sum(log(D^2)) ) /2
        dS = dlogdet*(inv(Us)/Ls).';
        dD = dlogdet./D;
        
        %S = RDL + eye(rank)
        dRDL = dS;
        
        %RDL = R.'*DL
        dDL = R*dRDL;
        dR = DL*dRDL.';
        
        % DL = bsxfun(@ldivide,D,L)
        dL = bsxfun(@ldivide,D,dDL);
        dD = dD - sum(DL.*dL,2);
    
    
    end

end

function test_this()

    dim = 5;
    rank = 2;
    
    D = randn(dim,1);
    R = randn(dim,rank);
    L = randn(dim,rank);
    
    M = diag(D) + L*R.';
    
    [log(abs(det(M))),logdetNice4(D,L,R)]
    
    
    testBackprop(@logdetNice4,{D,L,R},{1,1,1})
    

end