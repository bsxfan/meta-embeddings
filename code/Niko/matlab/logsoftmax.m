function [y,back] = logsoftmax(X,num)

    if nargin==0
        test_this();
        return;
    end

    [~,n] = size(X);

    [M,maxi] = max(X,[],1);
    Delta = bsxfun(@minus,X,M);
    expDelta = exp(Delta);
    arg = sum(expDelta,1);
    den = M + log(arg);
    y = sum(X(num),1) - sum(den,2);    
        
        
    back = @back_this;
        
    function dX = back_this(dy)
        dX = zeros(size(X));
        dX(num) = dy;
        dden = zeros(size(den));
        dden(:) = -dy;
        
        dM = dden;
        darg = dden./arg;
        
        dDelta = bsxfun(@times,expDelta,darg);
        
        dX = dX + dDelta;
        dM = dM - sum(dDelta,1);
        
        ii = sub2ind(size(X),maxi,1:n);
        dX(ii) = dX(ii) + dM;
        
        
    end


end


function test_this()

    m = 3;
    n = 5;
    X = randn(m,n);
    num = randi(m,1,n);
    testBackprop(@(X)logsoftmax(X,num),X);


end



