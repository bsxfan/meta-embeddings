function [y,back] = logsumexp(X)

    if nargin==0
        test_this();
        return;
    end

    mx = max(X,[],1);
    y = bsxfun(@plus,log(sum(exp(bsxfun(@minus,X,mx)),1)),mx);
        
        
    back = @back_this;
        
    function dX = back_this(dy)
        dX = bsxfun(@times,dy,exp(bsxfun(@minus,X,y)));
        
    end


end


function test_this()

    m = 3;
    n = 5;
    X = randn(m,n);
    testBackprop(@(X)logsumexp(X),X);


end



