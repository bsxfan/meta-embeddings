function [y,back] = SGME_MXE(A,B,D,As,Bs,labels,logPrior)

    if nargin==0
        test_this();
        return;
    end


    dA = zeros(size(A));
    dB = zeros(size(B));
    dD = zeros(size(D));
    dAs = zeros(size(As));
    dBs = zeros(size(Bs));

    
    
    [LEc,back1] = SGME_logexpectation(A,B,D);
    [LEs,back2] = SGME_logexpectation(As,Bs,D);

    dLEc = zeros(size(LEc));
    dLEs = zeros(size(LEs));
    

    m = length(LEs);   % #speakers
    n = length(LEc);   % #recordings

    scal = 1/(n*log(m+1));

    
    
    logPost = zeros(m+1,1);
    logPost(m+1) = logPrior(m+1);
    y = 0;
    for j=1:n
       AA = bsxfun(@plus,As,A(:,j)); 
       BB = bsxfun(@plus,Bs,B(:,j)); 
       [LEboth,back3] = SGME_logexpectation(AA,BB,D);  
       logPost(1:m) = logPrior(1:m) + LEboth.' - LEs.' - LEc(j); 
       [yj,back4] = sumlogsoftmax(logPost,labels(j));
       y = y - yj;
       
       
       dlogPost = back4(-1);
       dLEs = dLEs - dlogPost(1:m).';
       dLEc(j) = dLEc(j) - sum(dlogPost(1:m));
       dLEboth = dlogPost(1:m).';
       [dAA,dBB,dDj] = back3(dLEboth);
       dD = dD + dDj;
       dAs = dAs + dAA;
       dBs = dBs + dBB;
       dA(:,j) = sum(dAA,2);
       dB(:,j) = sum(dBB,2);
    end

    y = y*scal;

    back = @(dy) back_this(dy,dA,dB,dD,dAs,dBs);
    
    function [dA,dB,dD,dAs,dBs] = back_this(dy,dA,dB,dD,dAs,dBs)

        %[LEc,back1] = SGME_logexpectation(A,B,D);
        %[LEs,back2] = SGME_logexpectation(As,Bs,D).';
        [dA1,dB1,dD1] = back1(dLEc);
        [dAs2,dBs2,dD2] = back2(dLEs);
        dA = (dy*scal) * (dA + dA1);
        dB = (dy*scal) * (dB + dB1);
        dD = (dy*scal) * (dD + dD1 + dD2);
        dAs = (dy*scal) * (dAs + dAs2);
        dBs = (dy*scal) * (dBs + dBs2);
        
    end
    
    

end

function test_this()

    m = 3;
    n = 5;
    dim = 2;
    
    A = randn(dim,n);
    As = randn(dim,m);
    B = rand(1,n);
    Bs = rand(1,m);
    D = rand(dim,1);
    logPrior = randn(m+1,1);
    labels = randi(m,1,n);

    
    f = @(A,B,D,As,Bs) SGME_MXE(A,B,D,As,Bs,labels,logPrior);
    testBackprop(f,{A,B,D,As,Bs});



end

