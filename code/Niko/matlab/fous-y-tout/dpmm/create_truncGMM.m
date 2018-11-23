function model = create_truncGMM(W,F,alpha,m)
% This is a truncated version of DP micture model, with a specified maximum number of
% components. The observations are realted to the hidden cluster variables
% like in an SPLDA model. The hidden variable for cluster i is z_i in R^d.
% The observations, x_j are in R^D, where D>= d. The prior for the z_i is 
% IID: N(z_i | 0, I). The observations that belong to cluster i are 
% conditionally IID: N(x_j | F z_i, W^{-1} ). The SPLDA model parameters 
% are: 
%   F: D-by-d, factor loading matrix
%   W: D-by-D, within cluster precision (inverse covariance)
%
% The other parameters are for the symmetric Dirichlet prior on mixture 
% weights, which has parameters alpha and m, where m is the maximum number
% of mixture components: weights ~ Dir(alpha,m).
% More generally, alpha may be an m-by-1 vector, for a non-symmetric Dirichlet
% weight prior.

    if nargin==0
        test_this();
        return;
    end

    cholW = chol(W);
    dim = size(W,1);
    
    alpha = alpha(:);

    E = F'*W*F;  %meta-embedding precision (before diagonalization)
    [V,Lambda] = eig(E);  %E = V*Lambda*V';
    P = V.'*(F.'*W);  % projection to extract 1st-order meta-embedding stats
    Lambda = diag(Lambda);
    % The diagonal Lambda is the meta-embedding precision after
    % diagonalization.
    % We now have the likelihood, or meta-embedding:
    %  P(x | z) \propto exp[ z'Px - 1/2 z' Lambda z ], where z is the
    % hidden variable after diagonalization.
    %
    % The (normal) posterior for z, given n observations {x_j} has natural
    % parameters:
    %   sum_j P x_j and I + n Lambda
    
    FV = F*V; %projects from diagonalized Z to cluster means
    
    A = [];
    logThresh = log(1e-10);
    

    model.sampleData = @sampleData;
    model.sampleWeights = @sampleWeights;
    model.sampleZ = @sampleZ;
    model.sampleLabels = @sampleLabels;
    model.setData = @setData;
    model.label_log_posterior = @label_log_posterior;

    model.Means_given_labels = @Means_given_labels;
    model.fullGibbs_iteration = @fullGibbs_iteration;
    model.collapsedGibbs_iteration = @collapsedGibbs_iteration;
    model.mfvb_iteration = @mfvb_iteration;
    
    
    
    function setData(X)
        A = P*X;
    end
    
%     function Means = Z2Means(Z)
%         Means = FV*Z;
%     end
 

    %hlabels: m-by-n
    %Mu: D-by-m, posterior means for m cluster centroids
    %counts: 1-by-m, cluster occupancy counts (soft if hlabels is soft)
    %Q: posterior covariance for cluster i is: F*V*inv(diag(Q(:,i)))*V'*F.'
    function [Mu,Q,counts] = Means_given_labels(hlabels)
        if ~islogical(hlabels)
            [m,n] = size(hlabels);
            [~,L] = max(log(hlabels)-log(-log(rand(m,n))),[],1);
            hlabels = sparse(L,1:n,true,m,n);
        end
        counts = full(sum(hlabels,2)); %m-by-1
        Q = 1 + Lambda*counts.'; % d-by-m
        Zhat = (A*hlabels.') ./ Q; %d-by-m
        Mu = FV*Zhat;
    end
    
    
    % hlabels (one hot columns), m-by-n
    % A = P*X, d-by-n
    function Z = sampleZ(hlabels,counts)
        %counts = full(sum(hlabels,2)); %m-by-1
        Q = 1 + Lambda*counts.'; % d-by-m
        Zhat = (A*hlabels.') ./ Q; %d-by-m
        d = size(A,1);
        Z = Zhat + randn(d,m) ./ sqrt(Q);
    end


    function [weights,counts] = sampleWeights(hlabels)
        counts = sum(hlabels,2);
        weights = randDirichlet(alpha+counts,m,1);
    end



    % Z: d-by-m
    % A: d-by-n
    % weights: m-by-1
    function hlabels = sampleLabels(Z,weights)
        n = size(A,2);
        Gumbel = -log(-log(rand(m,n)));
        %ZLambdaZ = sum(Z.*bsxfun(@times,Lambda,Z),1);  % m-by-1
        ZLambdaZ = Lambda.'*Z.^2;  % m-by-1
        Score = bsxfun(@plus,log(weights)-ZLambdaZ.'/2,Z.'*A); %m-by-n
        [~,L] = max(Gumbel+Score,[],1);
        hlabels = sparse(L,1:n,true,m,n);
    end

    function hlabels = fullGibbs_iteration(hlabels)
        [weights,counts] = sampleWeights(hlabels);
        Z = sampleZ(hlabels,counts);
        hlabels = sampleLabels(Z,weights);
        
    end


    % hlabels (one hot columns), m-by-n
    % A = P*X, d-by-n
    function [Zhat,Q] = mfvb_Z(respbilties,counts)
        %counts = sum(respbilties,2); %m-by-1
        Q = 1 + Lambda*counts.'; % d-by-m
        Zhat = (A*respbilties.') ./ Q; %d-by-m
    end

    function [post_alpha,counts] = mfvb_Weights(respbilties)
        counts = sum(respbilties,2);
        post_alpha = alpha+counts;
    end

    % Z: d-by-m
    % A: d-by-n
    % weights: m-by-1
    function respbilties = mfvb_Labels(Zhat,Q,post_alpha)
        ZLambdaZ = Lambda.'*Zhat.^2 + sum(bsxfun(@rdivide,Lambda,Q),1);  % expected value
        log_weights = psi(post_alpha) - psi(sum(post_alpha)); % expected value
        R = bsxfun(@plus,log_weights-ZLambdaZ.'/2,Zhat.'*A); %m-by-n
        mx = max(R,[],1);
        R = bsxfun(@minus,R,mx);
        R(R<logThresh) = -inf;
        R = exp(R);
        R = bsxfun(@rdivide,R,sum(R,1));
        respbilties = R;
    end
    

    function respbilties = mfvb_iteration(respbilties)
        [post_alpha,counts] = mfvb_Weights(respbilties);
        [Zhat,Q] = mfvb_Z(respbilties,counts);
        respbilties = mfvb_Labels(Zhat,Q,post_alpha);
    end



    function logPrior = label_log_prior(counts)
    % Compute P(labels) by marginalizing over hidden weights. The output is 
    % in the form of un-normalized log probability. We use the 
    % candidate's formula:
    %    P(labels) = P(labels|weights) P(weights) / P(weights | labels)
    % where we conveniently set weights = 1/m. We ignore P(weights), 
    % because we do not compute the normalizer. Since weights are uniform,
    % we can also ignore P(labels|weights). We need to compute
    %   P(weights | labels) = Dir(weights | alpha + counts), with the
    % normalizer, which is a function of the counts (and the labels).
    
        logPrior = - logDirichlet(ones(m,1)/m,counts+alpha); % - log P(weights | labels)
    end


    function [llh,counts] = label_log_likelihood(hlabels)
        AL = A*hlabels.';  % d-by-m
        counts = sum(hlabels,2); %m-by-1
        Q = Lambda*counts.'; % d-by-m
        logdetQ = sum(log1p(Q),1);
        Q = 1 + Q; %centroid posterior precisions
        Mu = AL ./ Q; %centroid posterior means
        llh = (Mu(:).'*AL(:) - sum(logdetQ,2))/2;
    end

    function [logP,counts] = label_log_posterior(hlabels)
        if ~islogical(hlabels)
            [m,n] = size(hlabels);
            [~,L] = max(log(hlabels)-log(-log(rand(m,n))),[],1);
            hlabels = sparse(L,1:n,true,m,n);
        end
        [logP,counts] = label_log_likelihood(hlabels);
        logP = logP + label_log_prior(counts);
    end

    function hlabels = collapsedGibbs_iteration(hlabels)
        n = size(A,2);        
        for j=1:n
            hlabels(:,j) = false;
            AL0 = A*hlabels.';  % d-by-m
            counts0 = sum(hlabels,2); %m-by-1
            nB0 = Lambda*counts0.'; % d-by-m
            counts1 = counts0+1;
            AL1 = bsxfun(@plus,AL0,A(:,j));
            nB1 = bsxfun(@plus,nB0,Lambda);
            logdetQ0 = sum(log1p(nB0),1);
            logdetQ1 = sum(log1p(nB1),1);
            K0 = sum(AL0.^2./(1+nB0),1); 
            K1 = sum(AL1.^2./(1+nB1),1); 
            llh = (K1 - K0 - logdetQ1 + logdetQ0)/2;
            logPrior0 = gammaln(counts0+alpha);
            logPrior1 = gammaln(counts1+alpha);
            logPost = llh + (logPrior1 - logPrior0).';

            [~,c] = max(logPost - log(-log(rand(1,m))),[],2);
            hlabels(c,j) = true;
        end
        
    end

    function hlabels = collapsedGibbs_slow(hlabels)
        n = size(A,2);
        for j = 1:n
            logP = zeros(m,1);
            hlabels(:,j) = false;
            for i=1:m
                hlabels(i,j) = true;
                logP(i) = label_log_posterior(hlabels);
                hlabels(i,j) = false;
            end
            [~,c] = max(logP-log(-log(rand(m,1))),[],1);
            hlabels(c,j) = true;
        end
    end



    function [X,Means,Z,weights,hlabels] = sampleData(n)
        d = size(F,2);
        weights = randDirichlet(alpha,m,1);
        Z = randn(d,m);
        Means = F*Z; 
        [~,labels] = max(bsxfun(@minus,log(weights(:)),log(-log(rand(m,n)))),[],1);
        hlabels = sparse(labels,1:n,true,m,n);
        X = cholW\randn(dim,n) + Means*hlabels;
    end


    






end





function P = sampleP(dim,tame)
    R = rand(dim,dim-1)/tame;
    P = eye(dim) + R*R.';
end





function test_this()

    dim = 2;
    tame = 10;
    sep = 2;        %increase to move clusters further apart in simulated data
    
    small = false;
    
    alpha0 = 60;      %increase to get more clusters  
    
    if small
        n = 8;
        m = 20;
    else
        n = 1000;
        m = 100;
    end
    
    alpha = alpha0/m;
    
    
    W = sep*sampleP(dim,tame);
    B = sampleP(dim,tame);
    F = inv(chol(B));
    
    EER = testEER(W,F,1000)
    pause(4)    
    
    
    
    model = create_truncGMM(W,F,alpha,m);
    [X,Means,Z,weights,truelabels] = model.sampleData(n);
    
    counts = full(sum(truelabels,2).');
    nz = counts>0;
    nzcounts = counts(nz)
    
    
    
    close all;
    
    cg_labels = sparse(randi(m,1,n),1:n,true,m,n);  %random label init
    %cg_labels = sparse(ones(1,n),1:n,true,m,n);    %single cluster init
    
    bg_labels = cg_labels;
    mf_labels = cg_labels;
    model.setData(X);
    
    niters = 1000;
    cg_delta = zeros(1,niters);
    bg_delta = zeros(1,niters);
    mf_delta = zeros(1,niters);
    mft = 0;
    bgt = 0;
    cgt = 0;
    cg_wct = zeros(1,niters);
    bg_wct = zeros(1,niters);
    mf_wct = zeros(1,niters);
    
    oracle = model.label_log_posterior(truelabels);
    for i=1:niters
        
        tic;
        bg_labels = model.fullGibbs_iteration(bg_labels);
        bgt = bgt + toc;
        bg_wct(i) = bgt;
        
        tic;
        cg_labels = model.collapsedGibbs_iteration(cg_labels);
        cgt = cgt + toc;
        cg_wct(i) = cgt;
        
        tic;
        mf_labels = model.mfvb_iteration(mf_labels);
        mft = mft + toc;
        mf_wct(i) = mft;
        
        
        cg_delta(i) = model.label_log_posterior(cg_labels) - oracle;
        bg_delta(i) = model.label_log_posterior(bg_labels) - oracle;
        mf_delta(i) = model.label_log_posterior(mf_labels) - oracle;
        [bgMu,~,bg_counts] = model.Means_given_labels(bg_labels);
        [cgMu,~,cg_counts] = model.Means_given_labels(cg_labels);
        [mfMu,~,mf_counts] = model.Means_given_labels(mf_labels);
        bg_nz = bg_counts>0;
        cg_nz = cg_counts>0;
        mf_nz = mf_counts>0;
        subplot(2,2,1);plot(X(1,:),X(2,:),'.b',Means(1,nz),Means(2,nz),'*r',bgMu(1,bg_nz),bgMu(2,bg_nz),'*g');title('full Gibbs');
        subplot(2,2,2);plot(X(1,:),X(2,:),'.b',Means(1,nz),Means(2,nz),'*r',cgMu(1,cg_nz),cgMu(2,cg_nz),'*g');title('collapsed Gibbs');
        subplot(2,2,3);plot(X(1,:),X(2,:),'.b',Means(1,nz),Means(2,nz),'*r',mfMu(1,mf_nz),mfMu(2,mf_nz),'*g');title('mean field VB');
        subplot(2,2,4);semilogx(cg_wct(1:i),cg_delta(1:i),...
                            bg_wct(1:i),bg_delta(1:i),...
                            mf_wct(1:i),mf_delta(1:i));
        xlabel('wall clock');ylabel('log P(sampled labels) - log P(true labels)')                
        legend('clpsd Gibbs','full Gibbs','mfVB','Location','SouthEast');
        pause(0.1);
    end
    
    
    

end