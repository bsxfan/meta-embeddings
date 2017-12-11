function [W,Mu,TT] = train_TLDIvector(stats_or_ivectors,N,T,TT,nu,labels,niters,W,Mu)
% Inputs:
%   stats_or_ivectors:  can be either F, or ivectors
%       F: dm-by-n first-order stats (m: UBM size; d: feature dim; n: no segments)
%       ivectors: k-by-n, classical i-vector point-estimates
%   N: m-by-n zero order stats
%   T: dm-by-k factor loading matrix
%   TT: [optional] k^2-by-m, vectorized precomputed T_i'T_i, i=1:m
%   nu: scalar, nu>0, degrees of freedom
%   labels: 1-by-n, label vector, with label values in 1:L, where L is
%           number of languages.
%   niters: number of VBEM iterations to do 
%
%   W: k-by-k within class precision [optional, for initialization]
%   Mu: k-by-L language means        [optional, for initialization] 
%
%
% Outputs:
%   W: k-by-k within-class precision estimate
%   Mu: k-by-L class mean estimates


    if nargin==0
        test_this();
        return;
    end

    [A,B,k,n] = getPosteriorNatParams(stats_or_ivectors,N,T,TT);


    
    L = max(labels);
    if ~exist('Mu','var') || isempty(Mu)
        W = eye(k);
        Mu = zeros(k,L);
    else
        assert(all(size(W)==k));
        [k2,L2] = size(Mu);
        assert(k2==k && L==L2);
    end
    
    for iter=1:niters
        WMu = W*Mu;
        
        C = zeros(size(W));
        Pmeans = zeros(k,n);
        for ell=1:L
            tt = find(labels==ell);
            % E-step
            for t=tt
                Pt = W + reshape(B(:,t),k,k);  %posterior precision
                Pmean = Pt\(WMu(:,ell)+A(:,t)); %posterior mean
                Pmeans(:,t) = Pmean;
                C = C + inv(Pt);
            end
            
            %M-step
            D = Pmeans(:,tt);
            Mu(:,ell) = mean(D,2);
            D = bsxfun(@minus,D,Mu(:,ell));
            C = C + D*D.';
        end
        C = C/n;
        W = inv(C),
        Mu = Mu,
    
    end    
    


end



function test_this

  close all;

  %dimensions
  d = 10;        %feature dimension
  m = 10;        %no components
  k = 3;        %ivector dimension
  n = 1000;      %number of segments
  L = 2;        %number of languages
  mindur = 2;  
  maxdur = 100;
  niters = 3;
  
  T = randn(d*m,k);
  
  W = randn(k,k*2);
  W = W*W.'/k;

  
  UBM.logweights = randn(m,1)/5;
  UBM.Means = 5*randn(d,m);
  
  
  
  Mu = randn(k,L);
  
  [F,N,labels] = make_data(UBM,Mu,W,T,m,d,n,mindur,maxdur);
  dur = sum(N,1);
  L1 = labels==1;
  L2 = labels==2;

  
  TT = precomputeTT(T,d,k,m);
  
  ivectors = stats2ivectors(F,N,T,TT);
  
  
  LR1 = [1,-1]* score_LDIvector(F,N,T,TT,W,Mu);
  %LR2 = [1,-1]* score_LDIvector(ivectors,N,T,TT,W,Mu);

  subplot(4,1,1);plot(dur(L1),LR1(L1),'.r',dur(L2),LR1(L2),'.g');
  
  
  [W2,Mu2] = train_LDIvector(F,N,T,[],labels,niters);
  [W3,Mu3] = train_LDIvector(ivectors,N,T,[],labels,niters);
  [W4,Mu4,map] = train_standaloneLGBE(ivectors,labels);
  
  LR2 = [1,-1]* score_LDIvector(F,N,T,[],W2,Mu2);
  LR3 = [1,-1]* score_CPF(ivectors,N,T,TT,W3,Mu3);

  LR4 = [1,-1]*map(ivectors);
  subplot(4,1,2);plot(dur(L1),LR2(L1),'.r',dur(L2),LR2(L2),'.g');
  subplot(4,1,3);plot(dur(L1),LR3(L1),'.r',dur(L2),LR3(L2),'.g');
  subplot(4,1,4);plot(dur(L1),LR4(L1),'.r',dur(L2),LR4(L2),'.g');
  
end


function [F,N,labels,relConf] = make_data(UBM,Mu,W,T,m,d,n,mindur,maxdur)

  [k,L] = size(Mu);
  labels = randi(L,1,n);  
  Labels = sparse(labels,1:n,1,L,n);  %L-by-n one-hot class labels
  x = Mu*Labels+chol(W)\randn(k,n);   %i-vectors  
  Tx = T*x; 

  
  dur = randi(1+maxdur-mindur,1,n) + mindur -1;
  dm = d*m;
  F = zeros(dm,n);
  N = zeros(m,n);
  
  logweights = UBM.logweights;
  prior = exp(logweights-max(logweights));
  prior = prior/sum(prior);
  priorConfusion = exp(-prior.'*log(prior))-1;
  relConf = zeros(1,n);
  
  
  for i=1:n
      D = dur(i);
      states = randcatgumbel(UBM.logweights,D);
      States = sparse(states,1:D,1,m,D);
      X = (reshape(Tx(:,i),d,m)+UBM.Means)*States + randn(d,D);
      Q = bsxfun(@minus,UBM.Means.'*X,0.5*sum(X.^2,1));
      Q = bsxfun(@plus,Q,UBM.logweights-0.5*sum(UBM.Means.^2,1).');
      
      Q = exp(bsxfun(@minus,Q,max(Q,[],1)));
      Q = bsxfun(@rdivide,Q,sum(Q,1));
      CE = -(States(:).'*log(Q(:)))/D;  %cross entropy
      relConf(i) =  (exp(CE)-1)/priorConfusion;
      
      
      Ni = sum(Q,2);
      Fi = X*Q.';
      N(:,i) = Ni;
      F(:,i) = Fi(:);
      
  end
  



end



