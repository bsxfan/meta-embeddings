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
    GR = G*R;
    q = sum(R.*(GR),1);
    b = (nu + D - d)./(nu + q);
    
    n = b*labels.';
    
    
    [V,E] = eig(FWF);
    E = diag(E);
    
    nE = bsxfun(@times,n,E);
    VS = V'*S;
    nEVS = (1+nE).\VS;
    Mu = V*nEVS;  %posterior means
    
    WR = W*R;
    RWR = sum(R.*(WR),1);
    
    llh = ( Mu(:).'*S(:) - sum((nu+D)*log1p(RWR/nu),2) ) / 2;
    
    back = @back_this;
    
    function dR = back_this(dllh)
        
        % llh = ( Mu(:).'*S(:) - sum((nu+D)*log1p(RWR/nu),1) ) / 2
        dMu = (dllh/2)*S;
        dS = (dllh/2)*Mu;
        dRWR = (-dllh*(nu+D)/(2*nu))./(1+RWR/nu); 
        
        % RWR = sum(R.*(W*R),1)
        dR = bsxfun(@times,2*dRWR,WR);
        
        
        
        % Mu = V*nEVS
        dnEVS = V.'*dMu;
        
        % nEVS = (1+nE).\VS
        dVS = (1+nE).\dnEVS;
        dnE = -dVS.*nEVS;

        % VS = V'*S
        dS = dS + V*dVS;
        
        % nE = bsxfun(@times,n,E)
        dn = E.'*dnE;
        
        %n = b*labels.'
        db = dn*labels;
        
        % b = (nu + D - d)./(nu + q)
        dq = (-db.*b)./(nu+q);
        
        % q = sum(R.*(G*R),1)
        dR = dR + bsxfun(@times,2*dq,GR);
        
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
    nu = 2;
    f = @(R) htplda_llh(R,labels,F,W,nu);

    testBackprop(f,{R});


end
