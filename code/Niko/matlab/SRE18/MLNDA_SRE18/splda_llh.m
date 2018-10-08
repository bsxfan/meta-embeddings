function [llh,back] = splda_llh(R,labels,F,W)
% Inputs:
%   R: D-by-N, i-vectors
%   labels: sparse, logical K-by-N, one hot columns, K speakers, N recordings
    

    if nargin==0
        test_this();
        return;
    end

    FW = F.'*W;
    FWR = FW*R;
    S = FWR*labels.';  %first order natural parameter for speaker factor posterior
    n = full(sum(labels,2)');   %zero order stats (recordings per speaker)
    
    FWF = FW*F;
    [V,E] = eig(FWF);
    E = diag(E);
    
    nE = bsxfun(@times,n,E);
%    Mu = V*bsxfun(@ldivide,1+nE,V'*S);  %posterior means
    Mu = V*( (1+nE).\(V'*S) );  %posterior means
    

    RR = R*R.';
    llh = ( Mu(:).'*S(:) - RR(:).'*W(:) ) / 2;
    
    back = @back_this;
    
    function dR = back_this(dllh)
        
        dMu = (dllh/2)*S;
        dS = (dllh/2)*Mu;
        dRR = (-dllh/2)*W;
        dR = (2*dRR)*R;
        dS = dS + V*bsxfun(@ldivide,1+nE,V'*dMu);
        dFWR = dS*labels;
        dR = dR + FW.'*dFWR;
        
    end



end


function test_this()

    D = 4;
    d = 2;
    N = 10;
    K = 3;
    R = randn(D,N);
    labels = sparse(randi(K,1,N),1:N,true,K,N);
    F = randn(D,d);
    W = randn(D,D+1);W=W*W.';
    f = @(R) splda_llh(R,labels,F,W);

    testBackprop(f,{R});


end
