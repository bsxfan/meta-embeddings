function LLH = score_LDIvector(stats_or_ivectors,N,T,TT,W,Mu)
% stats_or_ivectors:  can be either F, or ivectors
%     F: dm-by-n first-order stats
%     ivectors: k-by-n, classical i-vector point-estimates
% N: m-by-n zero order stats
% T: dm-by-k factor loading matrix
% W: k-by-k within class precision
% Mu: k-by-L language means
% LLH: L-by-n language log-likelihoods



  
  
  
  [A,B,k,n] = getPosteriorNatParams(stats_or_ivectors,N,T,TT);
    
    
    [k2,L] = size(Mu); %L is number of languages
    assert(k2==k);
    assert(all(size(W)==k));
    
    WMu = W*Mu;
    offs = -0.5*sum(Mu.*WMu,1).';  %data-independent language offsets
    
    LLH = zeros(L,n);
    for t=1:n
        Bt = reshape(B(:,t),k,k);
        CWMu = (Bt+W)\WMu; %k-by-L
        LLH(:,t) = offs + 0.5*sum(CWMu.*WMu,1).' + CWMu.'*A(:,t);
    end
    
    
    


end