function PYCRP = create_PYCRP(alpha,beta,e,n)
% alpha: alpha>=0, concentration
% beta: 0<= beta <=1, discount


    

    if nargin==0
        test_this2();
        return;
    end

    
    if nargin==4
        PYCRP = create_PYCRP(1,0);
        PYCRP.set_expected_number_tables(e,n,alpha,beta);
        return;
    else
        assert(nargin==2);
    end
    
    

    assert(alpha>=0);
    assert(beta>=0 && beta<=1);

    
    PYCRP.logprob = @logprob;
    PYCRP.logprob3 = @logprob3;
    PYCRP.sample = @sample;
    PYCRP.expected_number_tables = @expected_number_tables;
    PYCRP.set_expected_number_tables = @set_expected_number_tables;
    PYCRP.ent = @ent;
    PYCRP.getParams = @getParams;
    
    function [concentration,discount] = getParams()
        concentration = alpha;
        discount = beta;
    end
    

    function logP = Gibbs(blocks)
        m,n = size(blocks);
        counts = sum(blocks,2);
        logP0 = logprob(counts);
        logP = zeros(m,n);
        for i=1:m+1
            tar = blocks(i,:);
            non = ~tar;
            
            counts_min = counts(labels(tar)) -1;
            logP(i,tar) = logP0 - gammaln(counts_min - beta);
            
            counts_plus = counts(labels(non))
        end
    end


    
    function e = expected_number_tables(n)
        e = ent(alpha,beta,n);
    end
        

    function e = ent(alpha,beta,n)
        if alpha==0 && beta==0
            e = 1;
        elseif isinf(alpha)
            e = n;
        elseif alpha>0 && beta>0      
            A = gammaln(alpha + beta + n) + gammaln(alpha + 1) ...
                - log(beta) - gammaln(alpha+n) - gammaln(alpha+beta);
            B = alpha/beta;
            e = B*expm1(A-log(B));  %exp(A)-B
        elseif alpha>0 && beta==0
            e = alpha.*( psi(n+alpha) - psi(alpha) );
        elseif alpha==0 && beta>0
            A = gammaln(beta + n) - log(beta) - gammaln(n) - gammaln(beta);
            e = exp(A);
        end
    end
    


    function [flag,concentration,discount] = set_expected_number_tables(e,n,concentration,discount)
        if ~isempty(concentration) && ~isempty(discount)
            error('you can''t specify both parameters');
        end
        if isempty(concentration) && isempty(discount)
            error('you must specify one parameter');
        end
        if e<1 || e>n
            error('expectation must be between 1 and %i',n);
        end
        
        if isempty(concentration)
            assert(discount>=0 && discount<1);
            beta = discount;
            if beta==0 && e==1
                alpha = 0;
                concentration = alpha;
                flag = 1;
                return;
            elseif e==n
                alpha = inf;
                concentration = alpha;
                flag = 1;
                return;
            end
        
            min_e = ent(0,beta,n);
            if e < min_e
                error('e=%g is impossible at discount=%g, minimum is e=%g',e,beta,min_e);
            end
            
            f = @(logalpha) ent(exp(logalpha),beta,n) - e;
            [logalpha,~,flag] = fzero(f,0);
            alpha = exp(logalpha);
            concentration = alpha;

        elseif isempty(discount)
        
            assert(concentration>=0);
            alpha = concentration;
            
            if alpha==0 && e==1
                beta = 0;
                discount = beta;
                flag = 1;
                return;
            elseif e==n
                beta = 1;
                discount = beta;
                flag = 1;
                return;
            end
        
            min_e = ent(alpha,0,n);
            if e < min_e
                error('e=%g is impossible at concentration=%g, minimum is e=%min_e',e,alpha,min_e);
            end
            
            f = @(logitbeta) ent(alpha,sigmoid(logitbeta),n) - e;
            [logitbeta,~,flag] = fzero(f,0);
            beta = sigmoid(logitbeta);
            discount = beta;
        
        end
    end

    function y = sigmoid(x)
        y = 1./(1+exp(-x));
    end

    
    function logP = logprob(counts)  %Wikipedia


        
        K = length(counts);
        T = sum(counts);
        
        if isinf(alpha) && beta==1 %singleton tables
            if all(counts==1)
                logP = 0;
            else
                logP = -inf;
            end
            return;
        end
        
        if alpha==0 && beta==0 %single table
            if K==1
                logP = 0;
            else
                logP = -inf;
            end
            return;
        end
        
        if alpha>0 && beta>0 % 2-param Pitman-Yor generalization
            logP = gammaln(alpha) - gammaln(alpha+T) + K*log(beta) ...
               + gammaln(alpha/beta + K) - gammaln(alpha/beta)  ...
               + sum(gammaln(counts-beta)) ...
               - K*gammaln(1-beta);
        elseif beta==0 && alpha>0 % classical CRP
            logP = gammaln(alpha) + K*log(alpha) - gammaln(alpha+T) + sum(gammaln(counts));
       
        elseif beta>0 && alpha==0
            logP = (K-1)*log(beta) + gammaln(K) - gammaln(T) ...
                   - K*gammaln(1-beta) + sum(gammaln(counts-beta));
        end
           
    end


    % Seems wrong
%     function logP = logprob2(counts)  % Goldwater
% 
% 
%         
%         K = length(counts);
%         T = sum(counts);
%         if beta>0  %Pitman-Yor generalization
%             logP = gammaln(1+alpha) - gammaln(alpha+T) ...
%                    + sum(beta*(1:K-1)+alpha) ...
%                    + sum(gammaln(counts-beta)) ...
%                    - K*gammaln(1-beta);
%         else %1 parameter CRP
%             logP = gammaln(1+alpha) + (K-1)*log(alpha) - gammaln(alpha+T) + sum(gammaln(counts));
%         end
%            
%     end


    % Agrees with Wikipedia version (faster for small counts)
    function logP = logprob3(counts)
        logP = 0;
        n = 0;
        for k=1:length(counts)
            % seat first customer at new table
            if k>1
                logP = logP +log((alpha+(k-1)*beta)/(n+alpha)); 
            end
            n = n + 1;
            % seat the rest at this table
            for i = 2:counts(k)
                logP = logP + log((i-1-beta)/(n+alpha));
                n = n + 1;
            end
        end
    end



    function [labels,counts] = sample(T)

        labels = zeros(1,T);
        counts = zeros(1,T);
        labels(1) = 1;
        K = 1; %number of classes
        counts(1) = 1;
        for i=2:T
            p = zeros(K+1,1);
%             for k=1:K
%                 p(k) = counts(k) - beta;
%             end
            p(1:K) = counts(1:K) - beta; 
            p(K+1) = alpha + K*beta;
            [~,k] = max(randgumbel(K+1,1) + log(p));  
            labels(i) = k;
            if k>K
                K = K + 1;
                assert(k==K);
                counts(k) = 1;
            else
                counts(k) = counts(k) + 1; 
            end
        end
        counts = counts(1:K);
        labels = labels(randperm(T));
        
        
    end
    
    
end



function test_this2()
    
    T = 20;
    e = 10;
    N = 1000;

    crp1 = create_PYCRP(0,[],e,T);
    [concentration,discount] = crp1.getParams()
    
    crp2 = create_PYCRP([],0,e,T);
    [concentration,discount] = crp2.getParams()
    

    K1 = zeros(1,T);
    K2 = zeros(1,T);
    for i=1:N
        [~,counts] = crp1.sample(T);
        K = length(counts);
        K1(K) = K1(K) + 1;

        [~,counts] = crp2.sample(T);
        K = length(counts);
        K2(K) = K2(K) + 1;
    end
    e1 = sum((1:T).*K1)/N
    e2 = sum((1:T).*K2)/N
    
    close all;
    
    subplot(2,1,1);bar(1:T,K1);
    subplot(2,1,2);bar(1:T,K2);
    
    K1 = K1/sum(K1);
    K2 = K2/sum(K2);
    %dlmwrite('K1.table',[(1:T)',K1'],' ');
    %dlmwrite('K2.table',[(1:T)',K2'],' ');
    
    for i=1:T
        fprintf('(%i,%6.4f) ',2*i-1,K1(i))
    end
    fprintf('\n');
    for i=1:T
        fprintf('(%i,%6.4f) ',2*i,K2(i))
    end
    fprintf('\n');
    
    
end


function test_this()

    alpha1 = 0.0;
    beta1 = 0.6;
    crp1 = create_PYCRP(alpha1,beta1);
    
    
    alpha2 = 0.1;
    beta2 = 0.6;
    crp2 = create_PYCRP(alpha2,beta2);
    
    
    close all;
    figure;
    hold;
    
    for i=1:100;
        L1 = crp1.sample(100);
        L2 = crp2.sample(100);
        C1=full(sum(int2onehot(L1),2));
        C2=full(sum(int2onehot(L2),2));
        x11 = crp1.logprob(C1);
        x12 = crp2.logprob3(C1);
        x22 = crp2.logprob3(C2);
        x21 = crp1.logprob(C2);
        
        plot(x11,x12,'.g');
        plot(x21,x22,'.b');
         
    end

    figure;hold;
    crp = crp1;
    for i=1:100;
        L1 = crp.sample(100);
        C1=full(sum(int2onehot(L1),2));
        x = crp.logprob(C1);
        y = crp.logprob3(C1);
        
        plot(x,y);
         
    end
     
     
     
     

end

