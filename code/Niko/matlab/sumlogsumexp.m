function [y,back] = sumlogsumexp(X)

    if nargin==0
        test_this();
        return;
    end

    mx = max(real(X),[],1);
    yy = mx + log(sum(exp(bsxfun(@minus,X,mx)),1));
    y = sum(yy,2);    
        
    back = @back_this;
        
    function dX = back_this(dy)
        dX = dy*exp(bsxfun(@minus,X,yy));
        
    end


end


function test_this()

    m = 3;
    n = 5;
    X = randn(m,n);
    testBackprop(@(X)sumlogsumexp(X),X);


end

