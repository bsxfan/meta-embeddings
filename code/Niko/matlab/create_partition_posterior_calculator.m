function calc = create_partition_posterior_calculator(prior,poi)
% Inputs:
%   prior: Exchangeable prior over partitions, for example CRP. It needs to
%          implement prior.logprob(counts), where counts are the number of 
%          customers per table (partition block sizes).
%   poi: partition of interest, given as an n-vector of table assignments,
%        where there are n customers. The tables are numbered 1 to m. 
    
    if nargin==0
        test_this();
        return;
    end

    n = length(poi);  %number of customers
    ns = 2^n-1;       %number of non-empty customer subsets
    subsets = logical(mod(fix(bsxfun(@rdivide,0:ns,2.^(0:n-1)')),2));
    subsets = subsets(:,2:end);  % dump empty subset

    
    function [weights,counts] = labels2weights(labels)
        [blocks,counts] = labels2blocks(labels);
        [tf,loc] = ismember(blocks',subsets','rows');
        assert(all(tf));
        weights = false(ns,1);
        weights(loc) = true; 
    end
    
    [poi_weights,counts] = labels2weights(poi);
    log_prior_poi = prior.logprob(counts);
    
    Bn = Bell(n);
    PI = create_partition_iterator(n);
    Weights = false(ns,Bn);
    log_prior = zeros(1,Bn);
    for j=1:Bn
        labels = PI.next();
        [Weights(:,j),counts] = labels2weights(labels);
        log_prior(j) = prior.logprob(counts);
    end
    
    
    
    calc.logPost = @logPost;
    calc.logPostPoi = @logPostPoi;
    
    
    function y = logPostPoi(A,B)
    % Inputs:
    %   A,B: n-column matrices of natural parameters for n meta-embeddings
    % Output:
    %   y: log P(poi | A,B, prior)
    
        [dim,n1] = size(A);
        assert(n1==n);
        assert(size(B,2)==n);


        % accumulate natural params for every subset
        A = A*subsets;  
        B = B*subsets;
        
        %compute subset likelihoods
        log_ex = zeros(1,ns); 
        for i=1:ns
            E = create_plain_GME(A(:,i),reshape(B(:,i),dim,dim),0);
            log_ex(i) = E.log_expectation();
        end
        
        num = log_prior_poi + log_ex*poi_weights;
        den = log_prior + log_ex*Weights;
        maxden = max(den);
        den = maxden+log(sum(exp(den-maxden)));
        y = num - den;
    
    end

    function f = logPost(A,B)
    % Inputs:
    %   A,B: n-column matrices of natural parameters for n meta-embeddings
    % Output:
    %   y: log P(poi | A,B, prior)
    
        [dim,n1] = size(A);
        assert(n1==n);
        assert(size(B,2)==n);


        % accumulate natural params for every subset
        A = A*subsets;  
        B = B*subsets;
        
        %compute subset likelihoods
        log_ex = zeros(1,ns); 
        for i=1:ns
            E = create_plain_GME(A(:,i),reshape(B(:,i),dim,dim),0);
            log_ex(i) = E.log_expectation();
        end
        
        llh = log_ex*Weights;
        den = log_prior + llh; 
        maxden = max(den);
        den = maxden+log(sum(exp(den-maxden)));
    
        function y = logpost_this(poi)
            [poi_weights,counts] = labels2weights(poi);
            log_prior_poi = prior.logprob(counts);
            num = log_prior_poi + log_ex*poi_weights;
            y = num - den;
        end
        
        f = @logpost_this;
        
    end



end

function test_this

    
    Mu = [-1 0 -1.1; 0 -3 0];
    C = [3 1 3; 1 1 1];
    A = Mu./C;
    B = zeros(4,3);
    B(1,:) = 1./C(1,:);
    B(4,:) = 1./C(2,:);
    scale = 3;
    B = B * scale;
    C = C / scale;
    
    
    close all;
    figure;hold;
    plotGaussian(Mu(:,1),diag(C(:,1)),'blue','b');
    plotGaussian(Mu(:,2),diag(C(:,2)),'red','r');
    plotGaussian(Mu(:,3),diag(C(:,3)),'green','g');
    axis('square');
    axis('equal');

    
    poi = [1 1 2];
    %prior = create_PYCRP(0,[],2,3);
    %prior = create_PYCRP([],0,2,3);
    
    create_flat_partition_prior(length(poi));
    
    calc = create_partition_posterior_calculator(prior,poi);
    f = calc.logPost(A,B);
    exp([f([1 1 2]), f([1 1 1]), f([1 2 3]), f([1 2 2]), f([1 2 1])])
    
    
end


