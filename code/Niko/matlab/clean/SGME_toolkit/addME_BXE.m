function [y,back] = addME_BXE(E,D,logexpectations,plo,wt,wn,tar)


    if nargin==0
        test_this();
        return;
    end
    
    n = size(E,2);    

    [LEc,back1] = logexpectations(E,D);
    y = 0;
    dE = zeros(size(E));
    dLEc = zeros(size(LEc));
    dD = zeros(size(D));
    for i=1:n-1
        jj = i+1:n;
        EE = bsxfun(@plus,E(:,i),E(:,jj));
        tari = full(tar(i,jj));
        [LE2,back2] = logexpectations(EE,D);
        llr = LE2 - LEc(i) - LEc(jj);
        
        arg_tar = -plo - llr(tari);
        noni = ~tari;
        arg_non = plo + llr(noni);
        
        y = y + wt*sum(softplus(arg_tar));
        y = y + wn*sum(softplus(arg_non));
        
        dllr = zeros(size(llr));
        dllr(tari) = (-wt)*sigmoid(arg_tar);
        dllr(noni) = wn*sigmoid(arg_non);
        
        dLE2 = dllr;
        dLEc(i) = dLEc(i) - sum(dllr);
        dLEc(jj) = dLEc(jj) - dllr;
        
        [dEE,dD2] = back2(dLE2);
        dD = dD + dD2;
        dE(:,i) = dE(:,i) + sum(dEE,2);
        dE(:,jj) = dE(:,jj) + dEE;
        
       
        
    end

    back = @(dy) back_this(dy,dE,dLEc,dD);
    
    function [dE,dD] = back_this(dy,dE,dLEc,dD)
        [dE1,dD1] = back1(dLEc);
        dE = dy*(dE + dE1);
        dD = dy*(dD + dD1);
    end
        
        
    




end


function y = sigmoid(x)
    y = 1./(1+exp(-x));
end

function y = softplus(x)
% y = log(1+exp(x));
    y = x;
    f = find(x<30);
    y(f) = log1p(exp(x(f)));
end



function test_this()

  zdim = 2;
  n = 5;
  A = randn(zdim,n);
  B = rand(1,n);
  E = [A;B];
  plo = randn;
  wt = rand;
  wn = rand;
  tar = sparse(randn(n)>0);
  D = rand(zdim,1);

  f = @(E,D) addME_BXE(E,D,@SGME_logexpectations,plo,wt,wn,tar);
  
  testBackprop(f,{E,D});
  
  
end




