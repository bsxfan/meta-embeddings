function [y,back] = SGME_logPsL(A,B,d,blocks,poi,num,logPrior)
        
    if nargin==0
        test_this();
        return;
    end

    
    m = size(blocks,1) - 1;

    At = A*blocks.';
    Bt = B*blocks.';
    [LEt,back1] = SGME_logexpectation(At,Bt,d);
    
    
    [LEc,back2] = SGME_logexpectation(A,B,d);
    
    Amin = At(:,poi) - A;
    Bmin = Bt(:,poi) - B;
    [LEmin,back3] = SGME_logexpectation(Amin,Bmin,d);
    
    LLR = zeros(size(blocks));
    for i=1:m

        tar = full(blocks(i,:));
        LLR(i,tar) = LEt(i) - LEmin(tar) - LEc(tar);
        
        non = ~tar;
        Aplus = bsxfun(@plus,A(:,non),At(:,i));
        Bplus = bsxfun(@plus,B(:,non),Bt(:,i));
        LLR(i,non) = SGME_logexpectation(Aplus,Bplus,d) - LEt(i) - LEc(non);
        
        
    end
    
    
    %y = LLR;
    [y,back5] = sumlogsoftmax(LLR + logPrior,num);
    
    
    
    back = @back_this;
    function [dA,dB,dd] = back_this(dy)
        dA = zeros(size(A));
        dB = zeros(size(B));
        dd = zeros(size(d));
        dLEt = zeros(size(LEt));
        dLEmin = zeros(size(LEmin));
        dLEc = zeros(size(LEmin));
        dAt = zeros(size(At));
        dBt = zeros(size(Bt));
        
        %[y,back5] = sumlogsoftmax(LLR + logPrior,num);
        dLLR = back5(dy);


        for k=1:m

            tar = full(blocks(k,:));
            %LLR(k,tar) = LEt(k) - LEmin(tar) - LEc(tar);
            row = dLLR(k,tar);
            dLEt(k) = dLEt(k) + sum(row);
            dLEmin(tar) = dLEmin(tar) - row;
            dLEc(tar) = dLEc(tar) - row;

            non = ~tar;
            Aplus = bsxfun(@plus,A(:,non),At(:,k));
            Bplus = bsxfun(@plus,B(:,non),Bt(:,k));
            %LLR(k,non) = SGME_logexpectation(Aplus,Bplus,d) - LEt(k) - LEc(non);
            [~,back4] = SGME_logexpectation(Aplus,Bplus,d);
            row = dLLR(k,non);
            [dAplus,dBplus,dd4] = back4(row);
            dLEt(k) = dLEt(k) - sum(row);
            dLEc(non) = dLEc(non) - row;
            dd = dd + dd4;
            dA(:,non) = dA(:,non) + dAplus;
            dB(:,non) = dB(:,non) + dBplus;
            dAt(:,k) = dAt(:,k) + sum(dAplus,2);
            dBt(:,k) = dBt(:,k) + sum(dBplus,2);
            
        end
        
        
        
        
        %[LEmin,back3] = SGME_logexpectation(Amin,Bmin,d);
        [dAmin,dBmin,dd3] = back3(dLEmin);
        dd = dd + dd3;
        
        %Amin = At(:,poi) - A;
        %Bmin = Bt(:,poi) - B;
        dA = dA - dAmin;
        dB = dB - dBmin;
        dAt = dAt + dAmin*blocks.';
        dBt = dBt + dBmin*blocks.';
        
        %[LEc,back2] = SGME_logexpectation(A,B,d);
        [dA2,dB2,dd2] = back2(dLEc);
        dA = dA + dA2;
        dB = dB + dB2;
        dd = dd + dd2;
        
        %[LEt,back1] = SGME_logexpectation(At,Bt,d);
        [dAt1,dBt1,dd1] = back1(dLEt);
        dAt = dAt + dAt1;
        dBt = dBt + dBt1;
        dd = dd + dd1;

        
        %At = A*blocks.';
        %Bt = B*blocks.';
        dA = dA + dAt*blocks;
        dB = dB + dBt*blocks;
    end



end

function test_this()

    em = 4;
    n = 7;
    dim = 2;
    
    prior = create_PYCRP([],0,em,n);
    poi = prior.sample(n);
    m = max(poi);
    blocks = sparse(poi,1:n,true,m+1,n);  
    num = find(blocks(:));    
    
    logPrior = prior.GibbsMatrix(poi);  

    d = rand(dim,1);
    A = randn(dim,n);
    b = rand(1,n);
    
    
    %f = @(A,b,d) SGME_logexpectation(A,b,d);
    %testBackprop(f,{A,b,d},{1,1,1});

    
    g = @(A,b,d) SGME_logPsL(A,b,d,blocks,poi,num,logPrior);
    testBackprop(g,{A,b,d},{1,1,1});




end

