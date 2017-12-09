function calc = create_pseudolikelihood_calculator(log_expectations,prior,poi)

    if nargin==0
        test_this();
        return;
    end

% Inputs:
%   log_expectations: function handle
%   poi: partition of interest, n-vector that maps customers to tables
%   prior: exchangeble prior, struct, for example CRP
%

    n = length(poi);  % # customers
    m = max(poi);     % # tables

    
    %(m+1)-by-n index matrix, with one-hot columns
    blocks = sparse(poi,1:n,true,m+1,n);  
    num = find(blocks(:));    
    
    
    %pre-allocate
    LLR = zeros(m+1,n);

    [logPrior,empties] = prior.GibbsMatrix(poi);  
    %logPrior = prior.slowGibbsMatrix(poi);
    
    
    calc.log_pseudo_likelihood = @log_pseudo_likelihood;
    

    function [y,back] = log_pseudo_likelihood(A,B)
        
        %original table stats
        At = A*blocks.';
        Bt = B*blocks.';
        %log-expectations for every original table
        [LEt,back1] = log_expectations(At,Bt);
        
        %log-expectations for every customer, alone at singleton table
        [LEc,back2] = log_expectations(A,B);  


        %For every customer, the stats for the table after that customer has
        %just left.
        Amin = bsxfun(@minus,At(:,poi),A);
        Bmin = bsxfun(@minus,Bt(:,poi),B);
        [LEmin,back3] = log_expectations(Amin,Bmin);
        
        
        for i=1:m
            tar = full(blocks(i,:));
            non = ~tar;
            
            %non-targets
            Aplus = bsxfun(@plus,A(:,non),At(:,i));
            Bplus = bsxfun(@plus,B(:,non),Bt(:,i));
            LLR(i,non) = log_expectations(Aplus,Bplus) - LEt(i) - LEc(non);
            
            %targets
            LLR(i,tar) = LEt(i) - LEmin(tar) - LEc(tar);
            
        end

        logPost = LLR + logPrior;
        [y,back4] = sumlogsoftmax(logPost,num);
        
        
        back = @back_this;
        
        
        function [dA,dB] = back_this(dy)

            %logPost = LLR + logPrior;
            %[y,back4] = sumlogsoftmax(logPost,num);
            dLLR = back4(dy);
            dLLR(end,empties) = 0;
            
            %do loop
            dA = zeros(size(A));
            dB = zeros(size(B));
            dLEt = zeros(1,m+1);
            dLEc = zeros(1,n);
            dLEmin = zeros(1,n);
            dAt = zeros(size(At));
            dBt = zeros(size(Bt));
            
            for k=1:m
                tar = full(blocks(k,:));
                non = ~tar;

                %targets
                %LLR(i,tar) = LEt(i) - LEmin(tar) - LEc(tar);
                temp = dLLR(k,tar);
                dLEt(k) = dLEt(k) + sum(temp);
                dLEmin(tar) = dLEmin(tar) - temp;
                dLEc(tar) = dLEc(tar) - temp;
                
                %non-targets 
                %(do some recomputation to save memory)
                Aplus = bsxfun(@plus,A(:,non),At(:,k));
                Bplus = bsxfun(@plus,B(:,non),Bt(:,k));
                [~,back5] = log_expectations(Aplus,Bplus);
                
                %LLR(i,non) = log_expectations(Aplus,Bplus) - LEt(i) - LEc(non);
                temp = dLLR(k,non);
                [dAplus,dBplus] = back5(temp);
                dLEt(k) = dLEt(k) - sum(temp);
                dLEc(non) = dLEc(non) - temp;
                
                %Aplus = bsxfun(@plus,A(:,non),At(:,i));
                %Bplus = bsxfun(@plus,B(:,non),Bt(:,i));
                dA(:,non) = dA(:,non) + dAplus;
                dB(:,non) = dB(:,non) + dBplus;
                dAt(:,k) = dAt(:,k) + sum(Aplus,2);
                dBt(:,k) = dBt(:,k) + sum(Bplus,2);

            end
            
            %[LEmin,back3] = log_expectations(Amin,Bmin);
            [dAmin,dBmin] = back3(dLEmin);
            
            %Amin = bsxfun(@minus,At(:,poi),A);
            %Bmin = bsxfun(@minus,Bt(:,poi),B);
            dA = dA - dAmin; 
            dB = dB - dBmin;
            dAt = dAt + dAmin*blocks.';
            dBt = dBt + dBmin*blocks.';
            
            %[LEc,back2] = log_expectations(A,B);  
            [dA2,dB2] = back2(dLEc);
            dA = dA + dA2;
            dB = dB + dB2;

            
            %[LEt,back1] = log_expectations(At,Bt);
            [dAt1,dBt1] = back1(dLEt);
            dAt = dAt + dAt1; 
            dBt = dBt + dBt1; 

            %At = A*blocks.';
            %Bt = B*blocks.';
            dA = dA + dAt*blocks;
            dB = dB + dBt*blocks;
            
        end
        
        
    
    end
    function [y,back] = log_pseudo_likelihood2(A,B,d)
        
        %original table stats
        At = A*blocks.';
        Bt = B*blocks.';
        %log-expectations for every original table
        [LEt,back1] = log_expectations(At,Bt);
        
        %log-expectations for every customer, alone at singleton table
        [LEc,back2] = log_expectations(A,B);  


        %For every customer, the stats for the table after that customer has
        %just left.
        Amin = bsxfun(@minus,At(:,poi),A);
        Bmin = bsxfun(@minus,Bt(:,poi),B);
        [LEmin,back3] = log_expectations(Amin,Bmin);
        
        
        for i=1:m
            tar = full(blocks(i,:));
            non = ~tar;
            
            %non-targets
            LLR(i,non) = SGME_LLR(At(:,i),A(:,non),Bt(:,i),B(:,non),d);
            
            %targets
            LLR(i,tar) = SGME_LLR(); LEt(i) - LEmin(tar) - LEc(tar);
            
        end

        logPost = LLR + logPrior;
        [y,back4] = sumlogsoftmax(logPost,num);
        
        
        back = @back_this;
        
        
        function [dA,dB] = back_this(dy)

            %logPost = LLR + logPrior;
            %[y,back4] = sumlogsoftmax(logPost,num);
            dLLR = back4(dy);
            dLLR(end,empties) = 0;
            
            %do loop
            dA = zeros(size(A));
            dB = zeros(size(B));
            dLEt = zeros(1,m+1);
            dLEc = zeros(1,n);
            dLEmin = zeros(1,n);
            dAt = zeros(size(At));
            dBt = zeros(size(Bt));
            
            for k=1:m
                tar = full(blocks(k,:));
                non = ~tar;

                %targets
                %LLR(i,tar) = LEt(i) - LEmin(tar) - LEc(tar);
                temp = dLLR(k,tar);
                dLEt(k) = dLEt(k) + sum(temp);
                dLEmin(tar) = dLEmin(tar) - temp;
                dLEc(tar) = dLEc(tar) - temp;
                
                %non-targets 
                %(do some recomputation to save memory)
                Aplus = bsxfun(@plus,A(:,non),At(:,k));
                Bplus = bsxfun(@plus,B(:,non),Bt(:,k));
                [~,back5] = log_expectations(Aplus,Bplus);
                
                %LLR(i,non) = log_expectations(Aplus,Bplus) - LEt(i) - LEc(non);
                temp = dLLR(k,non);
                [dAplus,dBplus] = back5(temp);
                dLEt(k) = dLEt(k) - sum(temp);
                dLEc(non) = dLEc(non) - temp;
                
                %Aplus = bsxfun(@plus,A(:,non),At(:,i));
                %Bplus = bsxfun(@plus,B(:,non),Bt(:,i));
                dA(:,non) = dA(:,non) + dAplus;
                dB(:,non) = dB(:,non) + dBplus;
                dAt(:,k) = dAt(:,k) + sum(Aplus,2);
                dBt(:,k) = dBt(:,k) + sum(Bplus,2);

            end
            
            %[LEmin,back3] = log_expectations(Amin,Bmin);
            [dAmin,dBmin] = back3(dLEmin);
            
            %Amin = bsxfun(@minus,At(:,poi),A);
            %Bmin = bsxfun(@minus,Bt(:,poi),B);
            dA = dA - dAmin; 
            dB = dB - dBmin;
            dAt = dAt + dAmin*blocks.';
            dBt = dBt + dBmin*blocks.';
            
            %[LEc,back2] = log_expectations(A,B);  
            [dA2,dB2] = back2(dLEc);
            dA = dA + dA2;
            dB = dB + dB2;

            
            %[LEt,back1] = log_expectations(At,Bt);
            [dAt1,dBt1] = back1(dLEt);
            dAt = dAt + dAt1; 
            dBt = dBt + dBt1; 

            %At = A*blocks.';
            %Bt = B*blocks.';
            dA = dA + dAt*blocks;
            dB = dB + dBt*blocks;
            
        end
        
        
    
    end







end






