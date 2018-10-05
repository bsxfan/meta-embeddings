function [f,fi,paramsz,fe] = create_nice_Trans(dim,rank)
% Creates affine transform, having a matrix: M = sigma I + L*D*L.', where
% L is of low rank and D is diagonal. The forward transform is:
%   f(X) = M \ X + offset 


    if nargin==0
        test_this();
        return;
    end


    paramsz = 1 + dim*rank + rank + dim;

    f = @f_this;
    fi = @fi_this;
    fe = @expand;
    

    function T = f_this(P,R)
        [sigma,L,d,offset] = expand(P);
        M = sigma*eye(dim) + L*diag(d)*L.';
        T = bsxfun(@plus,offset,M\R);
    end
    

    function [R,logdetJ,back] = fi_this(P,T)
        [sigma,L,d,offset,back0] = expand(P);
        
        Delta = bsxfun(@minus,T,offset);
        
        DL = bsxfun(@times,d,L.');
        DLDelta = DL*Delta;
        R = sigma*Delta + L*DLDelta;
        
        [logdet,back1] = logdetNice(sigma,L,d);
        n = size(T,2);
        logdetJ = -n*logdet;
        
        back = @back_that;
        
        function [dP,dT] = back_that(dR,dlogdetJ)
            
            %[logdetJ,back1] = logdetNice(sigma,L,d)
            [dsigma,dL,dd] = back1(-n*dlogdetJ);
            
            % R = sigma*Delta + L*DLDelta
            dsigma = dsigma + dR(:).'*Delta(:);
            dDelta = sigma*dR;
            dL = dL + dR*DLDelta.';
            dDLDelta = L.'*dR;
            
            % DLDelta = DL*Delta;
            dDL = dDLDelta*Delta.';
            dDelta = dDelta + DL.'*dDLDelta;
            
            % DL = bsxfun(@times,d,L.')
            dd = dd + sum(dDL.*L.',2);
            dL = dL + bsxfun(@times,dDL.',d');
            
            % Delta = bsxfun(@minus,T,offset)
            dT = dDelta;
            doffset = -sum(dDelta,2);
            
            dP = back0(dsigma,dL,dd,doffset);
        end
    end



    function [sigma,L,d,offset,back] = expand(P)
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
        
        sz = rank;
        d = P(at:at+sz-1);
        at = at + sz;
        
        sz = dim;
        offset = P(at:at+sz-1);
        at = at + sz;
        

        assert(at==length(P)+1);
        
        back = @back_this;
        
        function dP = back_this(dsigma,dL,dd,doffset)
            %dlogsigma = sigma*dsigma;
            %dP = [dlogsigma;dL(:);dd;doffset];
            
            %dP = [dsigma;dL(:);dd;doffset];

            dsqrtsigma = 2*dsigma*sqrtsigma;
            dP = [dsqrtsigma;dL(:);dd;doffset];
        
        
        end
        
        
        
        
    end



end


function test_this()

    dim = 5;
    rank = 2;
    n = 6;
    
    [f,fi,sz,fe] = create_nice_Trans(dim,rank);
    P = randn(sz,1);
    R = randn(dim,n);
    T = f(P,R);
    
    Ri = fi(P,T);
    test_inverse = max(abs(R(:)-Ri(:))),
    
    
    testBackprop_multi(fi,2,{P,T});
    %testBackprop_multi(fe,4,{P});
    
    
    


end

