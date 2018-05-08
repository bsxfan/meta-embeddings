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
    [D,T] = size(Emb);

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


    function part = create_partition(HL)
        
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
        
        %po = wrap(part);
        
    end

    function po = wrap(part)
        
        po.part = part;
        
        po.test_merge = @(i,j) test_merge(part,i,j);
        po.test_split = @(i,labels) test_split(part,i,labels);
        
        po.dumb_merge = @(i,j) dumb_merge(part,i,j);
        po.smart_merge = @(i,j) smart_merge(part,i,j);
        
        po.dumb_split = @(i,labels) dumb_split(part,i,labels);
        po.smart_split = @(i,labels) smart_split(part,i,labels);
        
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
        if isempty(labels) % no split 
            logPtilde = part.logPtilde;        
            state = [];
        else % split
            w = true(part.K,1);
            w(i) = false;
            state.counts = full(sum(labels,2));
            state.K = part.K + 1;
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


    function mpart = commit_merge(part,i,j,state)
        if i == j  %no merge
            mpart = part;
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
        
        mpart.llr = llh_subsets - llh_fine*HL.';
        
    end


    function spart = commit_split(part,i,labels,state)
        if isempty(labels) % no split
            spart = part;
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
        
        spart.llr = llh_subsets - llh_fine*HL.';
        
    end


    function [logQ,i,j] = dumb_merge(part,i,j)
        K = part.K;
        if nargin==1 || isempty(i)
            i = randi(K);
            j = randi(K);
        end
        logQ = -2*log(K);  %K^2 equiprobable states
    end

    function [logQ,i,j] = smart_merge(part,i,j)

        K = part.K;
        sample =  nargin==1 || isempty(i);

        if sample
            i = randi(K);
        end    
        logQ = -log(K);

        scounts = (part.counts(i)-beta) + part.counts;
        scounts(i) = part.counts(i);
        log_prior = gammaln(scounts).';
        lpi = log_prior(i);
        log_prior =  log_prior + CRP_Kterm(K-1);
        log_prior(i) = lpi + CRP_Kterm(K);

        sEi = part.PE(:,i);
        sE = bsxfun(@plus,part.PE(:,i),part.PE);
        sE(:,i) = sEi;
        log_post = log_prior + llhfun(sE);
        mx = max(log_post);
        norm = mx + log(sum(exp(log_post-mx)));
        if sample
            [~,j] = max(log_post + randgumbel(K,1) );
        end
        logQ = logQ + log_post(j) - norm;


    end


    function [logQ,i,labels] = dumb_split(part,i,labels)
        K = part.K;
        if nargin==1 || isempty(i)
            i = randi(K);          %K equiprobable choices for i
            n = part.counts(i);
            r = part.HL(i,:);
            kk = randi(2,1,n);
            labels = sparse(kk,find(r),true); % 1 + S(n,2) equiprobable states, 
                                         % where S(n,k) = 2^(n-1) - 1 is
                                         % the Stirling number of the 2nd
                                         % kind: the number of ways to
                                         % partition n items into 2
                                         % non-empty subsets. We add 1
                                         % because we allow one subset to
                                         % be empty.
                                         
            if min(sum(labels,2))==0  % one cluster is empty
                labels = []; %signal no split
            end
        else
            n = part.counts(i);
        end
        logQ = -log(K) - (n-1)*log(2);  % Q = (1/K) * ( S(n,2) + 1 )
    end

    function [logQ,i,labels] = smart_split(part,i,labels)
        K = part.K;
        
        % choose cluster to be split
        llh = -part.llh_subsets;     %alternative is to use llr
        sample = nargin==1 || isempty(i);
        if sample
            [~,i] = max( llh + randgumbel(1,K) );
        end
        mx = max(llh);
        norm = mx + log(sum(exp(llh-mx)));
        logQ = llh(i) - norm;
        
        %split
        n = part.counts(i);
        r = part.HL(i,:);
        
        if n==1 %no split
            if sample
                labels = [r;zeros(1,n)];
            end
            return;
        end
        
        E = Emb(:,r);
        
        %assign first one arbitrarily (no effect on Q)
        PE = [E(:,1),zeros(D,1)];
        counts = [1;0];
        nt = 1; %number of tables
        
        if sample
            kk = zeros(1,n);
            kk(1) = 1;
        else
            kk = ones(1,n);
            kk(labels(2,r)) = 2;
        end
        
        
        for j=2:n
            if nt == 1  % table 2 still empty
                logPrior = log([counts(1)-beta;nt*beta+alpha]);    
            else
                logPrior = log(counts-beta);
            end
            SE = bsxfun(@plus,E(:,j),PE);
            logPost = logPrior.' + llhfun(SE);
            mx = max(logPost);
            norm = mx + log(sum(exp(logPost-mx)));
            if sample
                [~,k] = max(logPost + randgumbel(1,2) );
            else
                k = kk(j);
            end
            logQ = logQ + logPost(k) - norm;
            PE(:,k) = SE(:,k);
            counts(k) = counts(k) + 1;
            if k==2
                nt = 2;
            end
        end
        
        if sample && nt==2
            labels = sparse(kk,find(r),true); 
        elseif sample && nt==1
            labels = [];  %signal no split
        end
        
        
    end



    function part1 = smart_split_dumb_merge(part0)
        logP0 = part0.logPtilde;
        if flip_coin() % split
            
            %fwd
            [logQ,i,labels] = smart_split(part0);
            logQfwd = logQ - log(2);
            [logP1,commit] = test_split(part0,i,labels);
            part1 = commit();
            
            %reverse
            [i,j] = merge_from_split(part,i,labels);
            logQrev = dumb_merge(part1,i,j) - log(2);

        else % merge
            
            %fwd
            [logQ,i,j] = dumb_merge(part0);
            logQfwd = logQ - log(2);
            [logP1,commit] = test_merge(part0,i,j);
            part1 = commit();
            
            %reverse
            [i,labels] = split_from_merge(i,j);
            logQrev = smart_split(part1,i,labels) - log(2);
        end
        
        MH = exp(logP1 - logP0 + logQrev - logQfwd);
        if rand > MH
            part1 = part0;  % reject
        end
        
        
    end



    function part1 = smart_merge_dumb_split(part0)
        logP0 = part0.logPtilde;
        if flip_coin() % split
            
            %fwd
            [logQ,i,labels] = dumb_split(part0);
            logQfwd = logQ - log(2);
            [logP1,commit] = test_split(part0,i,labels);
            part1 = commit();
            
            %reverse
            [i,j] = merge_from_split(part,i,labels);
            logQrev = smart_merge(part1,i,j) - log(2);

        else % merge
            
            %fwd
            [logQ,i,j] = smart_merge(part0);
            logQfwd = logQ - log(2);
            [logP1,commit] = test_merge(part0,i,j);
            part1 = commit();
            
            %reverse
            [i,labels] = split_from_merge(i,j);
            logQrev = dumb_split(part1,i,labels) - log(2);
        end
        
        MH = exp(logP1 - logP0 + logQrev - logQfwd);
        if rand > MH
            part1 = part0;  % reject
        end
        
        
    end





    function [i,j] = merge_from_split(part,i,labels)
        if isempty(labels) % no split
            j = i; % no merge
        else
            K = part.K;
            i = K;
            j = K+1;
        end
    end

    function [i,labels] = split_from_merge(part,i,j)
        if i==j % no split
            labels = [];
        else
            i = min(i,j);
            labels = part.HL([i,j],:);
        end
    end



end


function heads = flip_coin()
    heads = rand >= 0.5;
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

