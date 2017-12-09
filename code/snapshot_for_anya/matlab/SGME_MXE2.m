function [y,back] = SGME_MXE2(A,B,D,As,Bs,labels,logPrior)

    if nargin==0
        test_this();
        return;
    end


    dA = zeros(size(A));
    dB = zeros(size(B));
    dD = zeros(size(D));
    dAs = zeros(size(As));
    dBs = zeros(size(Bs));

    
    
    [LEs,back2] = SGME_logexpectation(As,Bs,D);

    dLEs = zeros(size(LEs));
    

    m = length(LEs);   % #speakers
    n = size(A,2);   % #recordings

    scal = 1/(n*log(m));

    
    
    y = 0;
    for j=1:n
       AA = bsxfun(@plus,As,A(:,j)); 
       BB = bsxfun(@plus,Bs,B(:,j)); 
       [LEboth,back3] = SGME_logexpectation(AA,BB,D);  
       logPost = logPrior + LEboth.' - LEs.'; 
       [yj,back4] = sumlogsoftmax(logPost,labels(j));
       y = y - yj;
       
       
       dlogPost = back4(-1);
       dLEs = dLEs - dlogPost.';
       dLEboth = dlogPost.';
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

        %[LEs,back2] = SGME_logexpectation(As,Bs,D).';
        [dAs2,dBs2,dD2] = back2(dLEs);
        dA = (dy*scal) * dA;
        dB = (dy*scal) * dB;
        dD = (dy*scal) * (dD + dD2);
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
    logPrior = randn(m,1);
    labels = randi(m,1,n);

    
    f = @(A,B,D,As,Bs) SGME_MXE2(A,B,D,As,Bs,labels,logPrior);
    testBackprop(f,{A,B,D,As,Bs});



end

