function [F,W,obj] = VB4HTPLDA_iteration_2(nu,F,W,R,labels,weights)
% Iteration of VB algorithm for HT-PLDA training. See HTPLDA_SGME_train_VB()
% for details. The model parameters F and W are updated. 
%
% Inputs:
%   nu: scalar, df > 0 (nu=inf is allowed: it signals G-PLDA)
%   F: D-by-d factor loading matrix, D > d
%   W: within-speaker precision, D-by-D, pos. def,
%   R: D-by-N i-vector data matrix (centered)
%   labels: M-by-N, one hot columns, labels for M speakers and N i-vectors
%           This is a large matrix and it is best represented as a sparse
%           logical matrix. 


    if nargin==0
        test_this;
        return;
    end

    scaling_mindiv = true;
    Z_mindiv = true;

    [D,d] = size(F);
    [M,N] = size(labels);
   
    
    % E-step
    
    P = F.'*W;         % d-by-D
    B0 = P*F;          % d-by-d common precision (up to scaling)
    [V,L] = eig(B0);   % eigendecomposition B0 = V*L*V'; L is diagonal, V is orthonormal  
    L = diag(L);
    VP = V.'*P;
    
    if isinf(nu)
        b = ones(1,N);
    else
        %G = W - P.'*(B0\P);
        G = W - VP.'*bsxfun(@ldivide,L,VP);   % inv(B0) = V*inv(L)*V'


        q = sum(R.*(G*R),1);
        b = (nu+D-d)./(nu+q);  %scaling         %1-by-N
    end
    if exist('weights','var') && ~isempty(weights)
        b = b.*weights;
    end
    
    bR = bsxfun(@times,b,R);
    S = bR*R.';                          % D-by-D  weighted 2nd-order stats 

    f = bR*labels.';                     % D-by-M  weighted 1st-order stats
    n = b*labels.';                      % 1-by-M  weighted 0-order stats
    tot_n = sum(n);
    
    logLPP = log1p(bsxfun(@times,n,L));  % d-by-M log eigenvalues of posterior precisions
    LPC = exp(-logLPP);                  % d-by-M eigenvalues of posterior covariances
    tracePC = sum(LPC,1);                % and the traces
    logdetPP = sum(logLPP,1);            % logdets of posterior precisions
    Z = V*(LPC.*(VP*f));                 % d-by-M posterior means
    T = Z*f.';                           % d-by-D
    
    R = bsxfun(@times,n,Z)*Z.' + V*bsxfun(@times,LPC*n(:),V.');
    C = ( Z*Z.' + V*bsxfun(@times,sum(LPC,2),V.') ) / M;
    
    
    
    logdetW = 2*sum(log(diag(chol(W))));
    logLH = (N/2)*logdetW + (D/2)*sum(log(b)) - 0.5*trprod(W,S) ...
             + trprod(T,P) -0.5*trprod(B0,R);  
    
    if isinf(nu)
        obj = logLH - KLGauss(logdetPP,tracePC,Z);
    else
        obj = logLH - KLGauss(logdetPP,tracePC,Z) - KLgamma(nu,D,d,b);
    end
    
    
    
    % M-step
    
    
    F = T.'/R;
    FT = F*T;
    
    if scaling_mindiv && true
       W = inv((S - (FT+FT.')/2)/tot_n);
    else
       W = inv((S - (FT+FT.')/2)/N);
    end
    
    
    
    CC = chol(C)';
    fprintf('  cov(z): trace = %g, logdet = %g\n',trace(C),2*sum(log(diag(CC))));
    if Z_mindiv
        F = F*CC;
    end;
    


end

function y = trprod(A,B)
    y = A(:).'*B(:);
end


function kl = KLGauss(logdets,traces,Means)
    d = size(Means,1);
    M = length(logdets);
    kl = ( sum(traces) - sum(logdets) + trprod(Means,Means) - d*M)/2;
    
end

function kl = KLgamma(nu,D,d,lambdahat)

    % prior has expected value a0/b0 = 1
    a0 = nu/2;
    b0 = nu/2;
    
    %Posterior
    a = (nu + D - d) / 2;    
    b = a ./ lambdahat;
    %This value for a is a thumbsuck: mean-field VB gives instead a =(nu+D)/2, 
    %while b is chosen to give lambdahat = a/b. 
    %This gives a larger variance (a/b^2) than the mean-field would, which
    %is probbaly a good thing, since mean-field is known to underestimate variances. 
    
    
    
    kl = sum(gammaln(a0) - gammaln(a) + a0*log(b/b0) + psi(a)*(a-a0) + a*(b0-b)./b);

end

function test_this()

    d = 2;
    D = 20;      %required: xdim > zdim
    nu = 3;         %required: nu >= 1, integer, DF
    fscal = 3;      %increase fscal to move speakers apart
    
    F0 = randn(D,d)*fscal;

    W0 = randn(D,D+1); W0 = W0*W0.';
    
%     HTPLDA = create_HTPLDA_extractor(F,nu,W);
%     [Pg,Hg,dg] = HTPLDA.getPHd(); 
    
    
    N = 5000;
    em = N/10;
    %prior = create_PYCRP(0,[],em,n);
    prior = create_PYCRP([],0,em,N);
    [R,Z,precisions,labels] = sample_HTPLDA_database(nu,F0,prior,N,W0);
    M = max(labels);
    labels = sparse(labels,1:N,true,M,N); 
    
    
    
    F = randn(D,d);
    W = eye(D);
    niters = 100;
    y = zeros(1,niters);
    for i=1:niters
        [F,W,obj] = VB4HTPLDA_iteration(nu,F,W,R,labels);
        fprintf('%i: %g\n',i,obj);
        y(i) = obj;
    end

    plot(y);
    

end



