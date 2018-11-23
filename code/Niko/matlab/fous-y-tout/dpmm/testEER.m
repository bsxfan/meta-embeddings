function EER = testEER(W,F,nspk)

    [D,d] = size(F);
    Z = randn(d,nspk);
    Enroll = F*Z + chol(W)\randn(D,nspk);
    Tar = F*Z + chol(W)\randn(D,nspk);
    Non = F*randn(d,nspk) + chol(W)\randn(D,nspk);
    
    E = F'*W*F;  %meta-embedding precision (before diagonalization)
    [V,Lambda] = eig(E);  %E = V*Lambda*V';
    P = V.'*(F.'*W);  % projection to extract 1st-order meta-embedding stats
    Lambda = diag(Lambda);
    Aenroll = P*Enroll;
    Atar = P*Tar;
    Anon = P*Non;
    
    logMEE = @(A,n) ( sum(bsxfun(@rdivide,A.^2,1+n*Lambda),1) - sum(log1p(n*Lambda),1) ) /2;
    score = @(A1,A2) logMEE(A1+A2,2) - logMEE(A1,1) - logMEE(A2,1);
    tar = score(Aenroll,Atar);
    non = score(Aenroll,Anon);
    EER = eer(tar,non);

end
