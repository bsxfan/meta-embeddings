function [y,back] = SGME_BXE(A,B,D,plo,wt,wn,tar)


    if nargin==0
        test_this();
        return;
    end
    
    n = size(A,2);    

    [LEc,back1] = SGME_logexpectation(A,B,D);
    y = 0;
    dA = zeros(size(A));
    dB = zeros(size(B));
    dLEc = zeros(size(LEc));
    dD = zeros(size(D));
    for i=1:n-1
        jj = i+1:n;
        AA = bsxfun(@plus,A(:,i),A(:,jj));
        BB = bsxfun(@plus,B(:,i),B(:,jj));
        tari = full(tar(i,jj));
        [LE2,back2] = SGME_logexpectation(AA,BB,D);
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
        
        [dAA,dBB,dD2] = back2(dLE2);
        dD = dD + dD2;
        dA(:,i) = dA(:,i) + sum(dAA,2);
        dB(:,i) = dB(:,i) + sum(dBB,2);
        dA(:,jj) = dA(:,jj) + dAA;
        dB(:,jj) = dB(:,jj) + dBB;
        
       
        
    end

    back = @(dy) back_this(dy,dA,dB,dLEc,dD);
    
    function [dA,dB,dD] = back_this(dy,dA,dB,dLEc,dD)
        [dA1,dB1,dD1] = back1(dLEc);
        dA = dy*(dA + dA1);
        dB = dy*(dB + dB1);
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
  plo = randn;
  wt = rand;
  wn = rand;
  tar = sparse(randn(n)>0);
  D = rand(zdim,1);

  f = @(A,B,D) SGME_BXE(A,B,D,plo,wt,wn,tar);
  
  testBackprop(f,{A,B,D});
  
  
end




