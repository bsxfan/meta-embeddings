function [y,back] = sumlogsoftmax(X,num)

    if nargin==0
        test_this();
        return;
    end

    [den,back1] = sumlogsumexp(X);
    y = sum(X(num)) - den;    
        
        
    back = @back_this;
        
    function dX = back_this(dy)
      dX = back1(-dy);        
      dX(num) = dX(num) + dy;  
    end


end


function test_this()

    m = 3;
    n = 5;
    X = randn(m,n);
    labels = randi(m,1,n);
    num = sub2ind(size(X),labels,1:n);
    testBackprop(@(X)sumlogsoftmax(X,num),X);


end



