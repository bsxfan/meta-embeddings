function [y,back] = addME_pseudoscore(E,w,logexpectations,blocks,poi,num,logPrior)
        
    if nargin==0
        test_this();
        return;
    end

    
    if isempty(blocks)
        m = max(poi);
        n = length(poi);
        blocks = sparse(poi,1:n,true,m+1,n);
        num = find(blocks(:));
    else
        m = size(blocks,1) - 1;
    end

    if isstruct(logPrior)  % then it is prior
        prior = logPrior;
        logPrior = prior.GibbsMatrix(poi);
    end
    
    
    Et = E*blocks.';
    [LEt,back1] = logexpectations(Et,w);
    
    
    [LEc,back2] = logexpectations(E,w);
    
    Emin = Et(:,poi) - E;
    [LEmin,back3] = logexpectations(Emin,w);
    
    LLR = zeros(size(blocks));
    for i=1:m

        tar = full(blocks(i,:));
        LLR(i,tar) = LEt(i) - LEmin(tar) - LEc(tar);
        
        non = ~tar;
        Eplus = bsxfun(@plus,E(:,non),Et(:,i));
        LLR(i,non) = logexpectations(Eplus,w) - LEt(i) - LEc(non);
        
        
    end
    
    
    %y = LLR;
    [logPsL,back5] = sumlogsoftmax(LLR + logPrior,num);
    y = -logPsL;
    
    
    back = @back_this;
    function [dE,dw] = back_this(dy)
        dE = zeros(size(E));
        dw = zeros(size(w));
        dLEt = zeros(size(LEt));
        dLEmin = zeros(size(LEmin));
        dLEc = zeros(size(LEmin));
        dEt = zeros(size(Et));
        
        %[y,back5] = sumlogsoftmax(LLR + logPrior,num);
        dLLR = back5(-dy);


        for k=1:m

            tar = full(blocks(k,:));
            %LLR(k,tar) = LEt(k) - LEmin(tar) - LEc(tar);
            row = dLLR(k,tar);
            dLEt(k) = dLEt(k) + sum(row);
            dLEmin(tar) = dLEmin(tar) - row;
            dLEc(tar) = dLEc(tar) - row;

            non = ~tar;
            Eplus = bsxfun(@plus,E(:,non),Et(:,k));
            %LLR(k,non) = logexpectations(Aplus,w) - LEt(k) - LEc(non);
            [~,back4] = logexpectations(Eplus,w);
            row = dLLR(k,non);
            [dEplus,dw4] = back4(row);
            dLEt(k) = dLEt(k) - sum(row);
            dLEc(non) = dLEc(non) - row;
            dw = dw + dw4;
            dE(:,non) = dE(:,non) + dEplus;
            dEt(:,k) = dEt(:,k) + sum(dEplus,2);
            
        end
        
        
        
        
        %[LEmin,back3] = logexpectations(Emin,w);
        [dEmin,dw3] = back3(dLEmin);
        dw = dw + dw3;
        
        %Emin = Et(:,poi) - E;
        dE = dE - dEmin;
        dEt = dEt + dEmin*blocks.';
        
        %[LEc,back2] = logexpectations(E,w);
        [dE2,dw2] = back2(dLEc);
        dE = dE + dE2;
        dw = dw + dw2;
        
        %[LEt,back1] = logexpectations(Et,w);
        [dEt1,dw1] = back1(dLEt);
        dEt = dEt + dEt1;
        dw = dw + dw1;

        
        %Et = E*blocks.';
        dE = dE + dEt*blocks;
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

    w = rand(dim,1);
    A = randn(dim,n);
    b = rand(1,n);
    E = [A;b];
    
    f = @SGME_logexpectations;
    
    g = @(E,w) addME_pseudoscore(E,w,f,blocks,poi,num,logPrior);
    testBackprop(g,{E,w});




end

