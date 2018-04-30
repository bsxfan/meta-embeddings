function part = create_partition(alpha,beta,llhfun,Emb,HL)
% Inputs:
%   prior: struct with various function handles
%   llhfun: maps M-by-K matrix of meta-embeddings to 1-by-N vector of
%           log-likelihoods
%   Emb: M-by-N matrix of additive meta-embeddings, for N recordings
%   HL: 'one-hot labels': sparse, logical, K-by-N matrix, with one-hot 
%       columns. Encodes K hyppothesized speakers for the N recordings.
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
    
    counts = sum(HL,2);
    K = sum(counts);
    llh_K = CRP_Kterm(K);
    PE = Emb*HL.';  % pooled embeddings
    llh_subsets = gammaln(counts-beta) + llhfun(PE);
    
    
    part.getLabels = @getLabels;
    part.merge = @merge;
    part.getPtilde = @getPtilde;
    part.getPtilde_merge = @getPtilde_merge;
    
    
    
    function labels = getLabels()
        labels = HL;
    end


    %unnormalized partition posterior
    function Ptilde = getPtilde()
        Ptilde = llh_K + sum(llh_subsets);
    end

    %tentative merge for MH acceptance test
    function [Ptilde,llh_mergedset] = getPtilde_merge(i,j)
        if i==j
            Ptilde = getPtilde();
            llh_mergedset = [];
        else
            w = true(K,1);
            w([i,j]) = false;
            llh_mergedset = gammaln(counts(i)+counts(j)-beta) + llhfun(PE(:,i)+PE(:,j));
            Ptilde = CRP_Kterm(K-1) + llh_mergedset + llh_subsets*w;
        end
    end
    

    %commit to merge
    function merge(i,j,llh_mergedset)
        if i ~= j  %merge
            
            K = K-1;
            llh_K = CRP_Kterm(K);
            
            k = min(i,j);     %put merged cluster here
            ell = max(i,j);   %delete this cluster
            
            counts(k) = counts(i) + counts(j);
            counts(ell) = [];
            
            PE(:,k) = PE(:,i) + PE(:,j);
            PE(:,ell) = [];
            
            if ~exist('llh_mergedset','var') || isempty(llh_mergedset) 
                llh_subsets(k) = gammaln(counts(k)-beta) + llhfun(PE(:,k));
            else
                llh_subsets(k) = llh_mergedset;
            end
            llh_subsets(ell) = [];
            
            HL(k,:) = HL(i,:) | HL(j,:);
            HL(ell,:) = [];
            
        end
        %else do nothing
    end


    function split(i,)



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