function pp = create_partition_posterior(alpha,beta,llhfun,Emb)


    if nargin==0
        test_this();
        return;
    end


    if isinf(alpha) || (beta==1) || (alpha==0 && beta==0)
        error('degenerate cases not handled');
    end
    if beta>0
        Kfactor = log(beta) - gammaln(1-beta);
    else
        Kfactor = log(alpha);
    end

  
    llh_fine = llhfun(Emb);   %may be useful for deciding which cluster to split
    T = length(llh_fine);

    % ignores all terms not dependent on K (i.e. terms dependent just on
    % alpha,beta and T)
    function C = CRP_Kterm(K)
        if beta>0 
            C = K*Kfactor + gammaln(alpha/beta + K);
        else 
            C = K*Kfactor;
        end
    end

    %unnormalized partition posterior
    function logPtilde = getlogPtilde(part)
        logPtilde = part.llh_K + sum(part.llh_counts) + sum(part.llh_subsets);
    end


    pp.create_partition = @create_partition;


    function po = create_partition(HL)
        
        if ~exist('HL','var') || isempty(HL)
            HL = logical(speye(T));
        end
        
        part.HL = HL;
        part.counts = full(sum(HL,2));
        part.K = length(part.counts);
        part.llh_K = CRP_Kterm(part.K);
        part.llh_counts = gammaln(part.counts-beta);

        part.PE = Emb*part.HL.';  % pooled embeddings
        part.llh_subsets = llhfun(part.PE);
        part.llr = part.llh_subsets - llh_fine*part.HL.';  % LLRs for each cluster: coarse vs fine
        part.logPtilde = getlogPtilde(part);
        
        po = wrap(part);
        
    end

    function po = wrap(part)
        
        po.part = part;
        po.test_merge = @(i,j) test_merge(part,i,j);
        po.test_split = @(i,labels) test_split(part,i,labels);
        po.dumb_merge = @(i,j) dumb_merge(part,i,j);
        po.dumb_split = @(i,labels) dumb_split(part,i,labels);
        
    end

    %tentative merge for MH acceptance test
    function [logPtilde,commit] = test_merge(part,i,j)
        if i==j
            logPtilde = part.logPtilde;
            state = [];
        else
            w = true(part.K,1);
            w([i,j]) = false;
            state.llh_counts = gammaln(part.counts(i)+part.counts(j)-beta);
            state.PE = part.PE(:,i) + part.PE(:,j);
            state.llh_subsets = llhfun(state.PE);
            state.llh_K = CRP_Kterm(part.K-1);
            logPtilde = state.llh_K + part.llh_counts.'*w + sum(state.llh_counts) ...
                                    + part.llh_subsets*w + sum(state.llh_subsets);
            state.logPtilde = logPtilde;
        end
        commit = @() commit_merge(part,i,j,state);
    end

    function [logPtilde,commit] = test_split(part,i,labels)
    % i: cluster to split
    % labels: one hot label matrix: 
    %   row indices are new cluster indices (starting at 1)
    %   column indices are original recording (embedding) indices  
        if min(sum(labels,2))==0  % no split 
            logPtilde = part.logPtilde;        
            state = [];
        else % split
            w = true(part.K,1);
            w(i) = false;
            state.counts = full(sum(labels,2));
            state.K = part.K - 1 + length(state.counts);
            state.llh_K = CRP_Kterm(state.K);
            state.PE = Emb*labels.';  
            state.llh_subsets = llhfun(state.PE);
            state.llh_counts = gammaln(state.counts-beta);
            logPtilde = state.llh_K + part.llh_counts.'*w + part.llh_subsets*w + ...
                        sum(state.llh_counts) + sum(state.llh_subsets);
            state.logPtilde = logPtilde;     
        end
        commit = @() commit_split(part,i,labels,state);
    end


    function po = commit_merge(part,i,j,state)
        if i == j  %no merge
            po = wrap(part);
            return;
        end
            
        mpart.K = part.K - 1;
        mpart.llh_K = state.llh_K;

        k = min(i,j);     %put merged cluster here
        ell = max(i,j);   %delete this cluster

        counts = part.counts;
        counts(k) = counts(i) + counts(j);
        counts(ell) = [];
        mpart.counts = counts;

        PE = part.PE;
        PE(:,k) = state.PE;
        PE(:,ell) = [];
        mpart.PE = PE;

        llh_subsets = part.llh_subsets;
        llh_subsets(k) = state.llh_subsets;
        llh_subsets(ell) = [];
        mpart.llh_subsets = llh_subsets;

        llh_counts = part.llh_counts;
        llh_counts(k) = state.llh_counts;
        llh_counts(ell) = [];
        mpart.llh_counts = llh_counts;

        HL = part.HL;
        HL(k,:) = HL(i,:) | HL(j,:);
        HL(ell,:) = [];
        mpart.HL = HL;

        mpart.logPtilde = state.logPtilde;
            
        po = wrap(mpart);
        
    end


    function po = commit_split(part,i,labels,state)
        if min(sum(labels,2))==0 % no split
            po = wrap(part);
            return;
        end
        
        HL = part.HL;
        HL(i,:) = [];
        HL = [HL;labels];  
        spart.HL = HL;
        
        PE = part.PE;
        PE(:,i) = [];
        PE = [PE,state.PE];
        spart.PE = PE;
        
        llh_subsets = part.llh_subsets;
        llh_subsets(i) = [];
        llh_subsets = [llh_subsets,state.llh_subsets];
        spart.llh_subsets = llh_subsets;
        
        llh_counts = part.llh_counts;
        llh_counts(i) = [];
        llh_counts = [llh_counts;state.llh_counts];
        spart.llh_counts = llh_counts;
        
        counts = part.counts;
        counts(i) = [];
        counts = [counts;state.counts];
        spart.counts = counts;
        

        spart.llh_K = state.llh_K;
        spart.K = state.K;
        spart.logPtilde = state.logPtilde;
        
        po = wrap(spart);
    end


    function [logQ,i,j] = dumb_merge(part,i,j)
        K = part.K;
        if nargin==1 || isempty(i)
            i = randi(K);
            j = randi(K);
        end
        logQ = -2*log(K);  %K^2 equiprobable states
    end

    function [logQ,i,labels] = dumb_split(part,i,labels)
        K = part.K;
        if nargin==1 || isempty(i)
            i = randi(K);          %K equiprobable choices for i
            n = part.counts(i);
            r = part.HL(i,:);
            N = length(r);
            jj = rand(1,N)>0.5;
            labels = [r & jj; r & ~jj];  %given i (and n): there are 2^n/2 equiprobable choices 
                                         %(labels has 2^n states, but we can swap the 2 rows, which halves the choices) 
        else
            n = part.counts(i);
        end
        logQ = -log(K) - (n-1)*log(2);  
    end





end



function test_this()

    dim = 1;
    N = 10;
    Emb = randn(dim,N);
    llhfun = @(Emb) zeros(1,size(Emb,2));
    
    alpha = pi;
    beta = 1/pi;
    
    
    pp = create_partition_posterior(alpha,beta,llhfun,Emb);
    
    part = pp.create_partition();
    counts = part.part.counts.'

    for iter=1:100
        if rand>0.5  % split
            [~,i,labels] = part.dumb_split([],[]);
            [~,commit] = part.test_split(i,labels);
            if rand>0.5 
                part = commit();
            end
        else %merge
            [~,i,j] = part.dumb_merge([],[]);
            [~,commit] = part.test_merge(i,j);
            if rand>0.5 
                part = commit();
            end
        end
        
        counts = part.part.counts.'
        
    end





end

