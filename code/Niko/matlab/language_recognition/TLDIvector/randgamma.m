function y = randgamma(a,b,n)
  % Returns a sample from Gamma(a, b). 
  % Algorithm:
  % G. Marsaglia and W.W. Tsang, A simple method for generating gamma
  % variables, ACM Transactions on Mathematical Software, Vol. 26, No. 3,
  % Pages 363-372, September, 2000.
  % http://portal.acm.org/citation.cfm?id=358414

% Edited by Niko
b = 1/b;  %change b from scale to rate parameter, to agree with Bishop  
if ~exist('n','var') || isempty(n)
    n=1;
end
  
y = zeros(1,n);
a0=a;
for i=1:n
    
  if a0 < 1
    % boost using Marsaglia's (1961) method: gam(a) = gam(a+1)*U^(1/a)
    boost = exp(log(rand())/a0);
    a = a0 + 1;
  else 
    a = a0;  
    boost = 1;
  end

  d = a-1.0/3; 
  c = 1.0/sqrt(9*d);
  
  while 1
    v = -1.0;
    while v <= 0
      x = randn();
      v = 1 + c*x;
    end
    v = v*v*v;
    x = x*x;
    u = rand();
    if (u < 1 - 0.0331*x*x) || (log(u) < 0.5*x + d*(1 - v + log(v)))
      break;
    end
  end % while 1
  y(i) = b*boost*d*v;

end
end % randgamma




