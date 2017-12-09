function TBE = create_T_backend(nu,dim,K)
% Create a (multivariate) T-distribution generative backend for multiclass classification.
% The classes have different means, but the scatter matrix and degrees of
% freedom are common to all clases.
%
% This object provides a method for supervised ML training (EM algorithm),
% as well as a method for scoring  at runtime (class log-likelihoods).
%
% Inputs:
%   nu: scalar >0, degrees of freedom
%   dim: data dimensionality
%   K: number of classes
%
% Typical usage:
%  > TBE = create_T-backend(nu,dim,K);  %nu is fixed by user and not learnt during training
%  > TBE.train(TrainData,L,10); % TrainData: dim-by-N, L: K-by-N, (sparse) one-hot labels
%  > LLH = TBE.logLH(TestData)  
%
% For EM algorithm, see: 
%   Geoffrey J. McClachlan and Thriyambakam Krishnan, The EM Algorithm and Extensions, 
%   2nd Ed. John Wiley & Sons, 2008. Section 2.6 EXAMPLE 2.6: MULTIVARIATE t-DISTRIBUTION WITH KNOWN
%   DEGREES OF FREEDOM


    if nargin==0
        test_this();
        return;
    end

    assert(nu>0);

    Mu = zeros(dim,K);
    C = eye(dim);
    R = [];
    RMu = [];
    muWmu = [];
    logdetC = 0;
    prepare();
    
    TBE.logLH = @logLH;
    TBE.getParams = @getParams;
    TBE.setParams = @setParams;
    TBE.train = @train;
    TBE.simulate = @simulate;
    TBE.randParams = @randParams;
    TBE.test_error_rate = @test_error_rate;
    TBE.cross_entropy = @cross_entropy;
    
    function [Mu1,C1] = getParams()
        Mu1 = Mu;
        C1 = C;
    end

    function setParams(Mu0,C0)
        Mu = Mu0;
        C = C0;
        prepare();
    end
    
    function [obj,XE] = train(X,L,niters)
        [d,N] = size(X); assert(d==dim);
        [k,n] = size(L); assert(k==K && n==N);
        

        obj = zeros(1,niters+1);
        obj_i = EM_objective(X,L);
        obj(1) = obj_i;
        
        doXE = nargout>=2;
        if doXE
            XE = zeros(1,niters+1);
            XE_i = cross_entropy(X,L);
            XE(1) = XE_i;
            fprintf('%i: %g, %g\n',0,obj_i,XE_i);
        else
            fprintf('%i: %g\n',0,obj_i);
        end
        
        for i=1:niters
            EM_iteration(X,L);
            obj_i = EM_objective(X,L);
            obj(i+1) = obj_i;
            if doXE
                XE_i = cross_entropy(X,L);
                XE(i+1) = XE_i;
                fprintf('%i: %g, %g\n',i,obj_i,XE_i);
            else
                fprintf('%i: %g\n',i,obj_i);
            end
        end
    end



    %Class log-likelihood scores, with all irrelevant constants omitted
    function LLH = logLH(X,df)   
    %inputs: 
    %  X: dim-by-N, data    
    %  df: [optional default df = nu], scalar, df>0, degrees of freedom parameter
    %
    %output:
    %  LLH: K-by-N, class log-likelihoods
    
        if ~exist('df','var') || isempty(df)
            df = nu;
        else
            assert(df>0);
        end
        Delta = delta(X);
        LLH = (-0.5*(df+dim))*log1p(Delta/df);
    end


    function prepare()
        R = chol(C);   % R'R = C and W = inv(C) = inv(R)*inv(R')
        RMu = R.'\Mu;  % dim-dy-K
        muWmu = sum(RMu.^2,1);  % 1-by-K
        logdetC = 2*sum(log(diag(R)));
    end


    function Delta = delta(X)   
    %input X: dim-by-N, data    
    %output Delta: K-by-N, squared Mahalanobis distances between data and means 
        RX = R.'\X;  % dim-by-N
        Delta = bsxfun(@minus,sum(RX.^2,1),(2*RMu).'*RX);  %K-by-N
        Delta = bsxfun(@plus,Delta,muWmu.');
    end


    function EM_iteration(X,L)
        Delta = sum(L.*delta(X),1);  %1-by-N
        u = (nu+dim)./(nu+Delta);  %1-by-N posterior expectations of hiddden precision scaling factors
        Lu = bsxfun(@times,L,u);   %K-by-N
        normLu = bsxfun(@rdivide,Lu,sum(Lu,2)); 
        newMu = X*normLu.';   %dim-by-K
        diff = X - newMu*L;
        newC = (bsxfun(@times,diff,u)*diff.')/sum(u);
        setParams(newMu,newC);
    end


    function obj = EM_objective(X,L)
    % X: dim-by-N, data
    % K: K-by-N, one-hot labels
    % obj: scalar
    
        LLH = logLH(X);
        
        N = size(X,2);
        obj = L(:).'*LLH(:) - (N/2)*logdetC ;
        
    end


    function randParams(ncov,muscale)
        assert(ncov>=dim);
        D = randn(dim,ncov);
        C = D*D.';
        setParams(zeros(dim,K),C);
        Mu = muscale*simulate(K);
        setParams(Mu,C);
    end

    function [X,L] = simulate(N,df,L)
        if ~exist('L','var') || isempty(L)
           L = sparse(randi(K,1,N),1:N,1,K,N);
        end
        if ~exist('df','var') || isempty(df)
            df = ceil(nu);
        end
        u = sum(randn(df,N).^2,1)/df; %  chi^2 with df dregrees of freedom, scaled so that <u>=1
        X = Mu*L + bsxfun(@rdivide,R.'*randn(dim,N),sqrt(u));
    end

    %assuming flat prior for now
    function e = test_error_rate(X,L)
        N = size(X,2);
        LLH = TBE.logLH(X);
        [~,labels] = max(LLH,[],1);
        Lhat = sparse(labels,1:N,1,K,N);
        e = 1-(L(:).'*Lhat(:))/N;
    end

    %assuming flat prior for now
    function e = cross_entropy(X,L,df)
        if ~exist('df','var') || isempty(df)
            df = nu;
        else
            assert(df>0);
        end
        LLH = TBE.logLH(X,df);
        P = exp(bsxfun(@minus,LLH,max(LLH,[],1)));
        P = bsxfun(@rdivide,P,sum(P,1));
        e = -mean(log(full(sum(L.*P,1))),2)/log(K);
    end



end


function test_this()

    close all;

    dim = 100; % data dimensionality
    K = 10; % numer of classes
    nu = 3; % degrees of freedom (t-distribition parameter)
    N = K*1000;
    
    %create test and train data
    TBE0 = create_T_backend(nu,dim,K);
    TBE0.randParams(dim,5/sqrt(dim));
    [X,L] = TBE0.simulate(N);
    [Xtest,Ltest] = TBE0.simulate(N);
    
    
    TBE = create_T_backend(nu,dim,K);
    [obj,XE] = TBE.train(X,L,20);
    subplot(1,2,1);plot(obj);title('error-rate');
    subplot(1,2,2);plot(XE);title('cross-entropy');

    
    train_error_rate = TBE.test_error_rate(X,L),

    test_error_rate = TBE.test_error_rate(Xtest,Ltest),
    
    
    train_XE = TBE.cross_entropy(X,L),
    test_XE = TBE.cross_entropy(Xtest,Ltest),
    
    df = [0.1:0.1:10];
    XE = zeros(2,length(df));
    for i=1:length(df)
        XE(1,i) = TBE.cross_entropy(X,L,df(i));
        XE(2,i) = TBE.cross_entropy(Xtest,Ltest,df(i));
    end
    figure;plot(df,XE(1,:),df,XE(2,:));
    grid;xlabel('df');ylabel('XE');
    legend('train','test');
    
    
    
end


