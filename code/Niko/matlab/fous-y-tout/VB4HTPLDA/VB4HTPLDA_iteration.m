function [F,W,obj] = VB4HTPLDA_iteration(nu,F,W,R,labels)
% Inputs:
%   nu: scalar, df > 0
%   F: D-by-d factor loading matrix, D > d
%   W: within-speaker precision, D-by-D, pos. def,
%   R: D-by-N i-vector data matrix (centered)
%   labels: M-by-N, one hot columns, labels for M speakers and N i-vectors

    scaling_mindiv = true;
    Z_mindiv = true;

    [D,d] = size(F);
    [M,N] = size(labels);
   
    
    % E-step
    
    P = F.'*W;   % d-by-D
    B0 = P*F;    %common precision (up to scaling)
    [V,L] = eig(B0);  % eigendecomposition B0 = V*L*V'; L is diagonal, V is orthonormal  
    L = diag(L);
    VP = V.'*P;
    
    %G = W - P.'*(B0\P);
    G = W - VP.'*bsxfun(@ldivide,L,VP);   % inv(B0) = V*inv(L)*V'
    
    
    q = sum(R.*G*R,1);
    b = (nu+D-d)./(nu+q);  %scaling         %1-by-N
    
    bR = bsxfun(@times,b,R);
    S = bR*R.';                          % D-by-D  weighted 2nd-order stats 

    f = bR*labels.';                     % D-by-M  weighted 1st-order stats
    n = b*labels.';                      % 1-by-M  weighted 0-order stats
    tot_n = sum(n);
    
    LPP = 1 + bsxfun(@times,n,L);        % d-by-M eigenvalues of posterior precisions
    LPC = 1./LPP;                        % d-by-M eigenvalues of posterior covariances
    logdetPP = log(prod(Lpost,1));       % logdets of posterior precisions
    tracePC = sum(LPC,1);                % and the traces
    Z = V*(Lpost.\(VP*f));               % d-by-M posterior means
    T = Z*f;                             % d-by-D
    
    R = bsxfun(@times,n,Z)*Z.' + V*bsxfun(@times,LPC*n(:),V.');
    C = ( Z*Z.' + V*bsxfun(@times,sum(LPC,2),V.') ) / M;
    
    
    
    logdetW = 2*sum(log(diag(chol(W))));
    logLH = (N/2)*logdetW + (D/2)*sum(log(b)) - 0.5*W(:).'*S(:) ...
             + T(:).'*P(:) -0.5*B0(:).'*R(:);  
    
    obj = logLH - KLGauss(logdetPP,tracePC,Z) - KLgamma(nu,D,d,b);
    
    
    
    % M-step
    
    
    F = T.'/R;
    FT = F*T;
    
    if scaling_mindiv && true
       W = inv((S - (FT+FT.')/2)/tot_n);
    else
       W = inv((S - (FT+FT.')/2)/N);
    end
    
    
    
    CC = chol(C)';
    fprintf('  cov(z): trace = %g, logdet = %g\n',trace(C),2*sum(log(diag(CC))));
    if Z_mindiv
        F = F*CC;
    end;
    


end


function kl = KLGauss(logdets,traces,Means)
    d = length(mu);
    M = length(logdets);
    kl = ( sum(traces) - sum(logets) + Means(:).'*Means(:) - d*M)/2;
    
end

function kl = KLgamma(nu,D,d,lambdahat)

    % prior has expected value a0/b0 = 1
    a0 = nu/2;
    b0 = nu/2;
    
    %Posterior
    a = (nu + D - d) / 2;    
    b = a ./ lambdahat;
    %This value for a is a thumbsuck: mean-field VB gives instead a =(nu+D)/2, 
    %while b is chosen to give lambdahat = a/b. 
    %This gives a larger variance (a/b^2) than the mean-field would, which
    %is probbaly a good thing, since mean-field is known to underestimate variances. 
    
    
    
    kl = sum(gammaln(a0) - gammaln(a) + a0*log(b/b0) + psi(a)*(a-a0) + a*(b0-b)./b);

end


