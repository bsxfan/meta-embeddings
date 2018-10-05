function [f,fi,paramsz,fe] = create_nice_Trans3(dim,rank)
% Creates affine transform, having a matrix: M = sigma I + L*R.', where
% L and R are of low rank. The forward transform is:
%   f(X) = M \ X + offset 


    if nargin==0
        test_this();
        return;
    end


    paramsz = 1 + 2*dim*rank + dim;

    f = @f_this;
    fi = @fi_this;
    fe = @expand;
    

    function T = f_this(P,X)
        [sigma,L,R,offset] = expand(P);
        M = sigma*eye(dim) + L*R.';
        T = bsxfun(@plus,offset,M\X);
    end
    

    function [X,logdetJ,back] = fi_this(P,T)
        [sigma,L,R,offset,back0] = expand(P);
        
        Delta = bsxfun(@minus,T,offset);
        RDelta = R.'*Delta; 
        X = sigma*Delta + L*RDelta;
        
        [logdet,back1] = logdetNice3(sigma,L,R);
        n = size(T,2);
        logdetJ = -n*logdet;
        
        back = @back_that;
        
        function [dP,dT] = back_that(dX,dlogdetJ)
            
            %[logdetJ,back1] = logdetNice3(sigma,L,R)
            [dsigma,dL,dR] = back1(-n*dlogdetJ);
            
            % X = sigma*Delta + L*RDelta;
            dsigma = dsigma + dX(:).'*Delta(:);
            dDelta = sigma*dX;
            dL = dL + dX*RDelta.';
            dRDelta = L.'*dX;
            
            % RDelta = R.'*Delta;
            dR = dR + Delta*dRDelta.';
            dDelta = dDelta + R*dRDelta;
            
            % Delta = bsxfun(@minus,T,offset)
            dT = dDelta;
            doffset = -sum(dDelta,2);
            
            dP = back0(dsigma,dL,dR,doffset);
        end
    end



    function [sigma,L,R,offset,back] = expand(P)
        at = 1;
        
        sz = 1;

        %logsigma = P(at);
        %sigma = exp(logsigma);
        
        %sigma = P(at);
        
        sqrtsigma = P(at);
        sigma = sqrtsigma^2;
        at = at + sz;
        
        sz = dim*rank;
        L = reshape(P(at:at+sz-1),dim,rank);
        at = at + sz;
        
        sz = dim*rank;
        R = reshape(P(at:at+sz-1),dim,rank);
        at = at + sz;
        
        sz = dim;
        offset = P(at:at+sz-1);
        at = at + sz;
        

        assert(at==length(P)+1);
        
        back = @back_this;
        
        function dP = back_this(dsigma,dL,dR,doffset)
            %dlogsigma = sigma*dsigma;
            %dP = [dlogsigma;dL(:);dd;doffset];
            
            %dP = [dsigma;dL(:);dd;doffset];

            dsqrtsigma = 2*dsigma*sqrtsigma;
            dP = [dsqrtsigma;dL(:);dR(:);doffset];
        
        
        end
        
        
        
        
    end



end


function test_this()

    dim = 5;
    rank = 2;
    n = 6;
    
    [f,fi,sz,fe] = create_nice_Trans3(dim,rank);
    P = randn(sz,1);
    X = randn(dim,n);
    T = f(P,X);
    
    Xi = fi(P,T);
    test_inverse = max(abs(X(:)-Xi(:))),
    
    
    testBackprop_multi(fi,2,{P,T});
    %testBackprop_multi(fe,4,{P});
    
    
    


end

