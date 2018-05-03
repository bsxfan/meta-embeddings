function part = create_partition(alpha,beta,llhfun,Emb,HL)
% Inputs:
%   prior: struct with various function handles
%   llhfun: maps M-by-K matrix of meta-embeddings to 1-by-N vector of
%           log-likelihoods
%   Emb: M-by-N matrix of additive meta-embeddings, for N recordings
%   HL: 'one-hot labels': sparse, logical, K-by-N matrix, with one-hot 
%       columns. Encodes K hypothesized speakers for the N recordings.
%       [Optional: default is finest partition: logical(speye(N))]


    if isinf(alpha) || (beta==1) || (alpha==0 && beta==0)
        error('degenerate cases not handled');
    end
    if beta>0
        Kfactor = log(beta) - gammaln(1-beta);
    else
        Kfactor = log(alpha);
    end



    [~,N] = size(Emb);
    if ~exist('HL','var') || isempty(HL)
        HL = logical(speye(N));
    end
    
    llh_fine = llhfun(Emb);   %may be useful for deciding which cluster to split

    counts = sum(HL,2);
    K = sum(counts);
    llh_K = CRP_Kterm(K);
    llh_counts = gammaln(counts-beta);
    
    PE = Emb*HL.';  % pooled embeddings
    llh_subsets = llhfun(PE);
    
    
    part.getLabels = @getLabels;
    part.test_merge = @test_merge;
    part.getlogPtilde = @getlogPtilde;
    part.getPtilde_merge = @getPtilde_merge;
    part.test_split = @test_split;
    
    
    
    function labels = getLabels()
        labels = HL;
    end


    %unnormalized partition posterior
    function logPtilde = getlogPtilde()
        logPtilde = llh_K + sum(llh_counts) + sum(llh_subsets);
    end

    %tentative merge for MH acceptance test
    function [logPtilde,commit] = test_merge(i,j)
        if i==j
            logPtilde = getlogPtilde();
            state = [];
        else
            w = true(K,1);
            w([i,j]) = false;
            state.llh_counts = gammaln(counts(i)+counts(j)-beta);
            state.PE = PE(:,i) + PE(:,j);
            state.llh_subsets = llhfun(state.PE);
            state.llh_K = CRP_Kterm(K-1);
            logPtilde = state.llh_K + llh_counts*w + state.llh_counts + llh_subsets*w + state.llh_subsets;
        end
        commit = @() commit_merge(i,j,state);
    end
    

    function commit_merge(i,j,state)
        if i ~= j  %merge
            
            K = K-1;
            llh_K = state.llh_K;
            
            k = min(i,j);     %put merged cluster here
            ell = max(i,j);   %delete this cluster
            
            counts(k) = counts(i) + counts(j);
            counts(ell) = [];
            
            PE(:,k) = state.PE;
            PE(:,ell) = [];
            
            llh_subsets(k) = state.llh_subsets;
            llh_subsets(ell) = [];
            
            llh_counts(k) = state.llh_counts;
            llh_counts(ell) = [];

            HL(k,:) = HL(i,:) | HL(j,:);
            HL(ell,:) = [];
            
        end
        %else do nothing
    end

    function [logPtilde,commit] = test_split(i,labels)
    % i: cluster to split
    % labels: one hot label matrix: 
    %   row indices are new cluster indices (starting at 1)
    %   column indices are original recording (embedding) indices  
        if size(labels,1)>1 
            state.counts = sum(labels,2);
            state.K = K-1 + sum(state.counts);
            state.llh_K = CRP_Kterm(state.K);
            state.PE = Emb*labels.';  
            state.llh_subsets = llhfun(state.PE);
            state.llh_counts = gammaln(scounts-beta);
            logPtilde = state.llh_K + sum(state.llh_counts) + sum(state.llh_subsets);
        else  %no split
            logPtilde = getlogPtilde();        
            state = [];
        end
        commit = @() commit_split(i,labels,state);
    end


    function commit_split(i,labels,state)
        if size(labels,1)==1 % no split
            return;
        end
        
        
        HL(i,:) = [];
        HL = [HL;labels];  
        
        PE(:,i) = [];
        PE = [PE,state.PE];
        
        llh_subsets(i) = [];
        llh_subsets = [llh_subsets,state.llh_subsets];
        
        llh_counts(i) = [];
        llh_counts = [llh_counts,state.llh_counts];
        
        counts(i) = [];
        counts = [counts;state.counts];

        llh_K = state.llh_K;
        K = state.K;
        
        
    end



    % ignores all terms not dependent on K (i.e. terms dependent just on
    % alpha,beta and T)
    function C = CRP_Kterm(K)
        if beta>0 
            C = K*Kfactor + gammaln(alpha/beta + K);
        else 
            C = K*Kfactor;
        end
    end




end