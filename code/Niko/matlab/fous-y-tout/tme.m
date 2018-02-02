function y = tme(z,mu,W,nu)

  if nargin==0
      test_this();
      return;
  end

  Delta = bsxfun(@minus,z,mu);
  q = sum(Delta.*(W*Delta),1);
  dim = length(mu);
  
  y = -(nu+dim)/2 * log1p(q/nu);


end

function test_this()

    dim = 1;
    W = 1;
    z = -10:0.01:10;
    mu1 = -5;
    mu2 = 5;
    
    nu = 1;
    y1 = tme(z,mu2,W,nu);
    y2 = tme(z,mu1,W,nu) 
    
    
    plot(z,y1,z,y2,z,y1+y2);
    


end