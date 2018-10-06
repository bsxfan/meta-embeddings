function [Ft,Wt,back] = adaptSPLDA(Fcols,Frows,Cfac,F,W)




    Ft = F + Fcols*Frows;
    
    % Wt = inv(Ct), Ct = inv(W) + Cfac*Cfac'
    % Wt = W - W*Cfac*inv(I + Cfac'*W*Cfac)*Cfac'*W
    
    WCfac = W*Cfac;
    S = eye(size(Cfac,2)) + WCfac.'*Cfac;
    
    cholS = chol(S);
    WCS = WCfac\cholS;
    Wt = W - WCS*WCS.';
    
    
    %[L,U] = lu(S);
    





end

