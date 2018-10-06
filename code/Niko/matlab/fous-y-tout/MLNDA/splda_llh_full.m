function [llh,back] = splda_llh_full(labels,F,Wfac,R,slow,facW)
% Like splda_llh(), but backpropagates into all of F,W and R
%
% Inputs:
%   R: D-by-N, i-vectors
%   labels: sparse, logical K-by-N, one hot columns, K speakers, N recordings
%   F,W: SPLA parameters
%   slow: [optional, default = false] logical, use slow = true to test derivatives
%
%   facW: [optional, default = true] logical:
%           true: W = Wfac*Wfac'
%           false: W = Wfac
%
% Outputs:
%   llh: total log density: log P(R | F,W,labels)
    

    if nargin==0
        test_this();
        return;
    end
    
    if ~exist('slow','var')
        slow = false;
    end
    
    if ~exist('facW','var')
        facW = true;
    end

    [nspeakers,ndata] = size(labels);
    
    if facW
        W = Wfac * Wfac.';
    else
        W = Wfac;
    end
    
    
    FW = F.'*W;
    FWR = FW*R;
    S = FWR*labels.';  %first order natural parameter for speaker factor posterior
    n = full(sum(labels,2)).';   %zero order stats (recordings per speaker)
    
    FWF = FW*F;

    if slow
        [pn,back1] = posteriorNorm_slow(S,FWF,n);
    else
        [pn,back1] = posteriorNorm_fast(S,FWF,n);
    end    

    RR = R*R.';

    if ~slow
        cholW = chol(W);
        logdetW = 2*sum(log(diag(cholW)));
    else
        [Lw,Uw] = lu(W);
        logdetW = sum(log(diag(Uw).^2))/2;
    end
    
    llh = ( ndata*logdetW - RR(:).'*W(:) ) / 2 - sum(pn,2);
    
    
    
      
        back = @back_this;
    
    function [dF,dWfac,dR] = back_this(dllh)
        
        % llh = ( ndata*logdetW - RR(:).'*W(:) ) / 2 - sum(pn,2)
        dlogdetW = ndata*dllh;
        dRR = (-dllh/2)*W;
        dW = (-dllh/2)*RR;
        dpn = repmat(-dllh,1,nspeakers);
        
        % 2*log(sum(diag(cholW)))
        if ~slow
            dW = dW + dlogdetW*(cholW\inv(cholW.'));
        else
            dW = dW + dlogdetW*(Lw.'\inv(Uw.')); 
        end
        
        % RR = R*R.'
        dR = (2*dRR)*R;
        
        % [pn,back1] = posteriorNorm_fast(S,FWF,n)
        [dS,dFWF] = back1(dpn);
        
        % FWF = FW*F
        dFW = dFWF*F.';
        dF = FW.'*dFWF;
        
        % S = FWR*labels.'
        dFWR = dS*labels;
        
        % FWR = FW*R;
        dFW = dFW + dFWR*R.';
        if nargout >= 3
            dR = dR + FW.'*dFWR;
        end
        
        
        % FW = F.'*W;
        dW = dW + F*dFW;
        dF = dF + W*dFW.';
        
        % W = Wfac * Wfac.';
        if facW
            dWfac = 2*dW*Wfac;
        else
            dWfac = dW;
        end
        
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
    %W = randn(D,D+1);W=W*W.';
    Wfac = randn(D,D+1);
    slow = true;
    f_slow = @(F,Wfac,R) splda_llh_full(labels,F,Wfac,R,slow);
    f_fast = @(F,Wfac,R) splda_llh_full(labels,F,Wfac,R);
    
    
    fprintf('test function value equality:\n');
    delta = abs(f_slow(F,Wfac,R)-f_fast(F,Wfac,R)),
    
    fprintf('test slow derivatives:\n');
    testBackprop(f_slow,{F,Wfac,R},{1,0,1});
    
    [~,back] = f_slow(F,Wfac,R);
    [dFs,dWfacs,dRs] = back(pi);
    
    [~,back] = f_fast(F,Wfac,R);
    [dFf,dWfacf,dRf] = back(pi);
    
    fprintf('compare fast and slow derivatives:\n');
    deltaF = max(abs(dFs(:)-dFf(:))),
    deltaW = max(abs(dWfacs(:)-dWfacf(:))),
    deltaR = max(abs(dRs(:)-dRf(:))),
    


end
