function [f,fi,paramsz,fe] = create_nice_Trans4(dim,rank)
% Creates affine transform, having a matrix: M = D + L*R.', where D is
% diagonal and L and R are of low rank. The forward transform is:
%   f(X) = M \ X + offset 


    if nargin==0
        test_this();
        return;
    end


    paramsz = dim + 2*dim*rank + dim;

    f = @f_this;
    fi = @fi_this;
    fe = @expand;
    

    function T = f_this(P,X)
        [D,L,R,offset] = expand(P);
        M = diag(D) + L*R.';
        T = bsxfun(@plus,offset,M\X);
    end
    

    function [X,logdetJ,back] = fi_this(P,T)
        [D,L,R,offset,back0] = expand(P);
        
        Delta = bsxfun(@minus,T,offset);
        RDelta = R.'*Delta; 
        X = bsxfun(@times,D,Delta) + L*RDelta;
        
        [logdet,back1] = logdetNice4(D,L,R);
        n = size(T,2);
        logdetJ = -n*logdet;
        
        back = @back_that;
        
        function [dP,dT] = back_that(dX,dlogdetJ)
            
            %[logdetJ,back1] = logdetNice4(D,L,R)
            [dD,dL,dR] = back1(-n*dlogdetJ);
            
            % X = bsxfun(@times,D,Delta) + L*RDelta
            dD = dD + sum(dX.*Delta,2);
            dDelta = bsxfun(@times,D,dX);
            dL = dL + dX*RDelta.';
            dRDelta = L.'*dX;
            
            % RDelta = R.'*Delta;
            dR = dR + Delta*dRDelta.';
            dDelta = dDelta + R*dRDelta;
            
            % Delta = bsxfun(@minus,T,offset)
            dT = dDelta;
            doffset = -sum(dDelta,2);
            
            dP = back0(dD,dL,dR,doffset);
        end
    end



    function [D,L,R,offset,back] = expand(P)
        at = 1;
        
        sz = dim;

        %logsigma = P(at);
        %sigma = exp(logsigma);
        
        %sigma = P(at);
        
        sqrtD = P(at:at+sz-1);
        D = sqrtD.^2;
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
        
        function dP = back_this(dD,dL,dR,doffset)
            %dlogsigma = sigma*dsigma;
            %dP = [dlogsigma;dL(:);dd;doffset];
            
            %dP = [dsigma;dL(:);dd;doffset];

            dsqrtD = 2*sqrtD.*dD;
            dP = [dsqrtD;dL(:);dR(:);doffset];
        
        
        end
        
        
        
        
    end



end


function test_this()

    dim = 5;
    rank = 2;
    n = 6;
    
    [f,fi,sz,fe] = create_nice_Trans4(dim,rank);
    P = randn(sz,1);
    X = randn(dim,n);
    T = f(P,X);
    
    Xi = fi(P,T);
    test_inverse = max(abs(X(:)-Xi(:))),
    
    
    testBackprop_multi(fi,2,{P,T});
    %testBackprop_multi(fe,4,{P});
    
    
    


end

