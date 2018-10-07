function [Ft,Wt,back] = adaptSPLDA(Fcols,Fscal,Cfac,F,W)

    if nargin==0
        test_this();
        return;
    end


    Frank = length(Fscal);
    


    Ft = [bsxfun(@times,F,Fscal), Fcols];
    
    % Wt = inv(Ct), Ct = inv(W) + Cfac*Cfac'
    % Wt = W - W*Cfac*inv(I + Cfac'*W*Cfac)*Cfac'*W
    
    WCfac = W*Cfac;
    S = eye(size(Cfac,2)) + WCfac.'*Cfac;
    
    [Adj,back1] = LinvSR(WCfac,S,WCfac.');
    Wt = W - (Adj + Adj.')/2;  %numerically symmetrize
    
    
    back = @back_this;
    
    
    function [dFcols,dFscal,dCfac,dF,dW] = back_this(dFt,dWt)
        
        % Wt = W - (Adj + Adj.')/2
        dAdj = -(dWt+dWt.')/2;
        dW = dWt;
    
        % [Wt0,back1] = LinvSR(WCfac,S,WCfac.')
        [dL,dS,dR] = back1(dAdj);
        dWCfac = dL + dR.'; 
        
        % S = eye(size(Cfac,2)) + WCfac.'*Cfac;
        dWCfac = dWCfac + Cfac*dS.';
        dCfac = WCfac*dS;
        
        %WCfac = W*Cfac;
        if nargout>=5
            dW = dW + dWCfac*Cfac.';
        end
        dCfac = dCfac + W.'*dWCfac;
        
        % Ft = [bsxfun(@times,F,Fscal), Fcols]
        dFcols = dFt(:,Frank+1:end);                    %OK
        dFscal = sum(dFt(:,1:Frank).*F,1);              %OK
        if nargout>=4
            dF = bsxfun(@times,dFt(:,1:Frank),Fscal);   %OK
        end
        
    end





end


function test_this()

    dim = 10;
    Frank = 2;
    extra = 3;
    F = randn(dim,Frank);
    Fcols = randn(dim,extra);
    Fscal = randn(1,Frank);
    W = randn(dim,dim);
    Cfac = randn(dim,4);
    
    f1 = @(Fcols,Fscal,Cfac,F,W) adaptSPLDA(Fcols,Fscal,Cfac,F,W);
    f2 = @(Fcols,Fscal,Cfac) adaptSPLDA(Fcols,Fscal,Cfac,F,W);
    

    testBackprop_multi(f1,2,{Fcols,Fscal,Cfac,F,W});
    testBackprop_multi(f2,2,{Fcols,Fscal,Cfac});

end

