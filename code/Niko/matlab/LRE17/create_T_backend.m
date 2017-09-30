function TBE = create_T_backend(nu,dim,K)
% Create a (multvariate) T-distribution generative backend for multiclass classification.
% The classes have different means, but the scatter matrix and dgerees of
% freedom are common to all clases.
%
% This object provides methods for an EM algorithms for ML (supervised) training,
% as well as a runtime scoring method.
%
%
% For EM algorithm, see: 
%   Geoffrey J. MacLachlan and Thriyambakam Krishnan, The EM Algorithm and Extensions, 
%   2nd Ed. Jognn Wiley & Sons, 2008. Section 2.6 EXAMPLE 2.6: MULTIVARIATE t-DISTRIBUTION WITH KNOWN
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
    
    function [Mu1,C1] = getParams()
        Mu1 = Mu;
        C1 = C;
    end

    function setParams(Mu0,C0)
        Mu = Mu0;
        C = C0;
        prepare();
    end
    
    function obj = train(X,L,niters)
        [d,N] = size(X); assert(d==dim);
        [k,n] = size(L); assert(k==K && n==N);
        
        obj = zeros(1,niters+1);
        obj_i = EM_objective(X,L);
        obj(1) = obj_i;
        fprintf('%i: %g\n',0,obj_i);
        for i=1:niters
            EM_iteration(X,L);
            obj_i = EM_objective(X,L);
            obj(i+1) = obj_i;
            fprintf('%i: %g\n',i,obj_i);
        end
    end



    %Class log-likelihood scores, with all irrelevant constants omitted
    function LLH = logLH(X)   
    %input X: dim-by-N, data    
    %output LLH: K-by-N, class log-likelihoods 
        Delta = delta(X);
        LLH = (-0.5*(nu+dim))*log1p(Delta/nu);
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


end


function test_this()

    close all;

    dim = 3; % data dimensionality
    K = 10; % numer of classes
    nu = 1; % degrees of freedom (t-distribition parameter)
    N = K*1000;
    L = sparse(randi(K,1,N),1:N,1,K,N);  %K-by-N one-hot label matrix
    
    Mu = 3*randn(dim,K)  % class means
    R = randn(dim,dim);
    C = R*R'             % within-class scatter 
    u = sum(randn(nu,N).^2,1)/nu; %  chi^2 with nu dregrees of freedom, scaled so that <u>=1
    X = Mu*L + bsxfun(@rdivide,R*randn(dim,N),sqrt(u));

    
    TBE = create_T_backend(nu,dim,K);
    obj = TBE.train(X,L,20);
    plot(obj);

    [hatMu,hatC] = TBE.getParams(),
    
    
end


