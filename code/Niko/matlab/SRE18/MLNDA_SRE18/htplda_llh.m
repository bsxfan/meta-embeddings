function [llh,back] = htplda_llh(R,labels,F,W,nu)
% Inputs:
%   R: D-by-N, i-vectors
%   labels: sparse, logical K-by-N, one hot columns, K speakers, N recordings
    

    if nargin==0
        test_this();
        return;
    end

    [D,d] = size(F);
    
    FW = F.'*W;
    FWR = FW*R;
    
    S = FWR*labels.';  %first order natural parameter for speaker factor posterior
    %n = full(sum(labels,2)');   %zero order stats (recordings per speaker)
    
    FWF = FW*F;

    G = W - FW.'*(FWF\FW);
    q = sum(R,G*R,1);
    b = (nu + D - d)./(nu + q);
    
    n = b*labels.';
    
    
    [V,E] = eig(FWF);
    E = diag(E);
    
    nE = bsxfun(@times,n,E);
    VS = V'*S;
    nEVS = bsxfun(@ldivide,1+nE,VS);
    Mu = V*nEVS;  %posterior means
    
    RWR = sum(R.*(W*R),1);
    
    llh = ( Mu(:).'*S(:) - (nu+D)*log1p(RWR/nu) ) / 2;
    
    back = @back_this;
    
    function dR = back_this(dllh)
        
        % llh = ( Mu(:).'*S(:) - (nu+D)*log1p(RWR/nu) ) / 2
        dMu = (dllh/2)*S;
        dS = (dllh/2)*Mu;
        dRWR = (-dllh*(nu+D)/(2*nu))./(1+RWR/nu); 
        
        % RWR = sum(R.*(W*R),1)
        dR = bsxfun(@times,2*dRWR,R);
        
        
        
        % Mu = V*nEVS
        dNEVS = V.'*dMu;
        
        % nEVS = bsxfun(@ldivide,1+nE,VS)
        dVS = bsxfun(@ldivide,1+nE,dNEVS);
        dnE = -dVS*nEVS.';

        % VS = V'*S
        dS = dS + V*dVS;
        
        % nE = bsxfun(@times,n,E)
        dn = E.'*dnE;
        
        %n = b*labels.'
        db = dn*labels;
        
        % b = (nu + D - d)./(nu + q)
        dq = (-db.*b)./(nu+q);
        
        % q = sum(R,G*R,1)
        dR = dR + bsxfun(@times,2*dq,R);
        
        % S = FWR*labels.'
        dFWR = dS*labels;
        
        % FWR = FW*R;
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
