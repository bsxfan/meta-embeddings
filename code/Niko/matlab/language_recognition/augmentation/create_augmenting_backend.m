function ABE = create_augmenting_backend(nu,dim,T,K,L)
% Inputs:
%   nu: scalar nu>0, t-distribution degrees of freedom
%   dim: ivector dimension
%   T: i-vector extr\zctor T-matrix
%   K: UBM size
%   L: number of languages


    if nargin==0
        test_this();
        return;
    end

    assert(dim==size(T,2));
    
    TBE = create_T_backend(nu,dim,L);

    augment = [];
    
    
    
    ABE.train = @train;
    ABE.logLH = @logLH;
    ABE.test_error_rate = @test_error_rate;
    ABE.cross_entropy = @cross_entropy;
    
    function [obj,AX] = train(X,Z,Labels,niters,ntiters)
    % X: ivectors
    % Z: zero-order stats
    % Labels: sparse one-hot label matrix
        if ~exist('niters','var') || isempty(niters)
            niters = 1;
        end
        if ~exist('ntiters','var') || isempty(ntiters)
            ntiters = 10;
        end
        
        assert(size(Labels,1)==L);
        assert(size(Labels,2)==size(X,2));
        assert(size(Labels,2)==size(Z,2));
        assert(size(Z,1)==K);
        assert(size(X,1)==dim);
        
        AX = X;
        obj = [];
        for i=1:niters
            obj_i = TBE.train(AX,Labels,ntiters); % starts with parameters from prev. iteration
            obj = [obj(:);obj_i(:)];
            [Mu,C] = TBE.getParams();
            augment = augment_i_vectors(T,K,Mu,C);
            if i<niters || nargout>=2
                AX = augment(X,Z);
            end
        end
    
    end


    function [LLH,X] = logLH(X,Z)
        if exist('Z','var') && ~isempty(Z)
            X = augment(X,Z);
        end
        LLH = TBE.logLH(X); 
    end

    %assuming flat prior for now
    function e = test_error_rate(X,Z,Labels)
        N = size(X,2);
        LLH = logLH(X,Z);
        [~,labels] = max(LLH,[],1);
        Lhat = sparse(labels,1:N,1,L,N);
        e = 1-(Labels(:).'*Lhat(:))/N;
    end

    %assuming flat prior for now
    function e = cross_entropy(X,Z,Labels)
        LLH = logLH(X,Z);
        P = exp(bsxfun(@minus,LLH,max(LLH,[],1)));
        P = bsxfun(@rdivide,P,sum(P,1));
        e = -mean(log(full(sum(Labels.*P,1))),2)/log(L);
    end



end


function test_this()

    big = true;

    if big
        L = 10; %languages
        K = 1024; %UBM size
        nu = 2; %df
        dim = 400; %ivector dim
        fdim = 40; % feature dim
        minDur = 3*100;   %3 sec
        maxDur = 30*100;  %30 sec

        M = randn(dim,L);
        T = randn(K*fdim,dim);
        RR = randn(dim,2*dim);W = RR*RR';

        Ntrain = 100;
        Ntest = 100;
    else
        L = 3; %languages
        K = 10; %UBM size
        nu = 2; %df
        dim = 40; %ivector dim
        fdim = 5; % feature dim
        minDur = 3*100;   %3 sec
        maxDur = 30*100;  %30 sec

        M = randn(dim,L);
        T = randn(K*fdim,dim);
        RR = randn(dim,2*dim);W = RR*RR'/100;

        Ntrain = 100;
        Ntest = 100;
    end
    
    fprintf('generating data\n');
    [F,trainZ,trainLabels] = rand_ivector(M,nu,W,2,K,T,minDur,maxDur,Ntrain);
    [trainX,TT] = stats2ivectors(F,trainZ,T);
    [F,testZ,testLabels] = rand_ivector(M,nu,W,2,K,T,minDur,maxDur,Ntest);
    testX = stats2ivectors(F,testZ,T,TT);
    F = [];
    
    fprintf('training\n');
    ABE = create_augmenting_backend(nu,dim,T,K,L);
    ABE.train(trainX,trainZ,trainLabels,2);
    
    train_error_rate = ABE.test_error_rate(trainX,trainZ,trainLabels),
    test_error_rate = ABE.test_error_rate(testX,testZ,testLabels),
    
    train_XE = ABE.cross_entropy(trainX,trainZ,trainLabels),
    test_XE = ABE.cross_entropy(testX,testZ,testLabels),
    
    
end

