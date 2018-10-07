function [llh,back] = splda_llh_full(labels,F,W,R,slow)
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
    
    [nspeakers,ndata] = size(labels);
    
    
    
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

%     if ~slow
%         cholW = chol(W);
%         logdetW = 2*sum(log(diag(cholW)));
%     else
%         [Lw,Uw] = lu(W);
%         logdetW = sum(log(diag(Uw).^2))/2;
%     end

    [logdetW,back2] = logdetLU(W); 
    
    llh = ( ndata*logdetW - RR(:).'*W(:) ) / 2 - sum(pn,2);
    
    
    
      
    back = @back_this;
    
    
    
    function [dF,dW,dR] = back_this(dllh)
        
        % llh = ( ndata*logdetW - RR(:).'*W(:) ) / 2 - sum(pn,2)
        dlogdetW = ndata*dllh/2;
        if nargout>=3
            dRR = (-dllh/2)*W;
        end
        dW = (-dllh/2)*RR;
        dpn = repmat(-dllh,1,nspeakers);
        
        % [logdetW,back2] = logdetLU(W)
        dW = dW + back2(dlogdetW);
        
        
        % RR = R*R.'
        if nargout>=3
            dR = (2*dRR)*R;
        end
        
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
    slow = true;
    f_slow = @(F,W,R) splda_llh_full(labels,F,W,R,slow);
    f_fast = @(F,W,R) splda_llh_full(labels,F,W,R);
    f0 = @(F,W) splda_llh_full(labels,F,W,R);
    
    fprintf('test function value equality:\n');
    delta = abs(f_slow(F,W,R)-f_fast(F,W,R)),
    
    fprintf('test slow derivatives:\n');
    testBackprop(f_slow,{F,W,R},{1,1,1});
    
    %return
    
    [~,back] = f_slow(F,W,R);
    [dFs,dWs,dRs] = back(pi);
    
    [~,back] = f_fast(F,W,R);
    [dFf,dWf,dRf] = back(pi);
    
    [~,back] = f0(F,W);
    [dF0,dW0] = back(pi);    
    
    fprintf('compare fast and slow derivatives:\n');
    deltaF = max(abs(dFs(:)-dFf(:))),
    deltaW = max(abs(dWs(:)-dWf(:))),
    deltaR = max(abs(dRs(:)-dRf(:))),
    
    deltaF0 = max(abs(dFs(:)-dF0(:))),
    deltaW0 = max(abs(dWs(:)-dW0(:))),


end
