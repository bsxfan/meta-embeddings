function P = PLDA_mixture_responsibilities(w,F,W,R)

    if nargin==0
        P = test_this();
        return
    end

    K = length(w);
    
    if iscell(F)
        [D,d] = size(F{1});
    else
        [D,d] = size(F);
    end
    N = size(R,2);

    P = zeros(K,N);
    
    Id = eye(d);

    for k=1:K
        if iscell(F)
            Fk = F{k};
        else
            Fk = F;
        end
        Wk = W{k};
        Bk = Fk.'*Wk*Fk;
        Gk = Wk - Wk*Fk*((Id+Bk)\Fk.'*Wk);
        
        RGR = sum(R.*(Gk*R),1);
        logdetW = 2*sum(log(diag(chol(Wk))));
        logdetIB = 2*sum(log(diag(chol(Id+Bk))));
        
        P(k,:) = log(w(k)) + (logdetW - logdetIB - RGR)/2;
        
    end    

    
    P = exp(bsxfun(@minus,P,max(P,[],1)));
    P = bsxfun(@rdivide,P,sum(P,1));
    


end

function P = test_this()

    close all;

    d = 100;
    D = 400;
    N = 1000;
    
    K = 5;
    w = ones(1,K)/K;
    W = cell(1,K);
    W{1} = eye(D);
    for k=2:K
        W{k} = 2*W{k-1};
    end
    
    %F = randn(D,d);
    F = cell(1,K);
    for k=1:K
        F{k} = randn(D,d);
    end

    
    Z = randn(d,N*K);
    R = randn(D,N*K);
    jj = 1:N;
    for k=1:K
        R(:,jj) = F{k}*Z(:,jj) + chol(W{k})\randn(D,N);
        jj = jj + N;
    end
    
    P = PLDA_mixture_responsibilities(w,F,W,R);
    plot(P');




end



