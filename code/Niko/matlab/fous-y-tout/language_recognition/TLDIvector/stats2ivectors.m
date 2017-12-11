function [ivectors,TT,A,B] = stats2ivectors(F,Z,T,TT)
% F: dm-by-n first-order stats
% Z: m-by-n zero order stats
% T: dm-by-k factor loading matrix
% W: k-by-k within class precision
% Mu: k-by-L language means
% LLH: L-by-n language log-likelihoods
%
%
% ivectors: k-by-n, classical i-vector point-estimates


    if ~exist('TT','var')
        TT = [];
    end
    
    [A,B,k,n,TT] = getPosteriorNatParams(F,Z,T,TT);
   
    I = eye(k);
    ivectors= zeros(k,n);
    for t=1:n
        Bt = reshape(B(:,t),k,k);
        ivectors(:,t) = (I+Bt)\A(:,t);
    end
    
    
    
    
end
