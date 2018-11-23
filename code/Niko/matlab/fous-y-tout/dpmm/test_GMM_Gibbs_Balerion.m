function test_GMM_Gibbs_Balerion()

    close all;

    dim = 500;
    tame = 10;
    sep = 0.6e-3;        %increase to move clusters further apart in smulated data
    %sep = sep*200;
    %sep = 1;
    
    
    alpha0 = 1000;      %increase to get more clusters  
    
    n = 100000;
    m = 2000;
    
    alpha = alpha0/m;
    
    
    W = sep*sampleP(dim,tame);
    %B = sampleP(dim,tame);
    %F = inv(chol(B));
    d = 100;
    F = randn(dim,d);
    
    EER = testEER(W,F,1000)
    
    pause(4)
    
    
    
    
    model = create_truncGMM(W,F,alpha,m);
    [X,Means,Z,weights,truelabels] = model.sampleData(n);
    
    counts = full(sum(truelabels,2).');
    nz = counts>0;
    nzcounts = counts(nz)
    
    
    
    close all;
    
    labels = sparse(randi(m,1,n),1:n,true,m,n);  %random label init
    %labels = sparse(ones(1,n),1:n,true,m,n);    %single cluster init
    
    model.setData(X);
    
    niters = 2000;
    delta = zeros(1,niters);
    wct = 0;
    time = delta;
    
    
    oracle = model.label_log_posterior(truelabels);
    for i=1:niters
        
        tic;labels = model.fullGibbs_iteration(labels);wct = wct+toc;
        time(i) = wct;
        %labels = model.collapsedGibbs_iteration(labels);
        %labels = model.mfvb_iteration(labels);
        delta(i) = model.label_log_posterior(labels) - oracle;
        counts = full(sum(labels,2).');
        fprintf('%i: delta = %g, clusters = %i\n',i,delta(i),sum(counts>0));
    end
    
    
    labels = sparse(randi(m,1,n),1:n,true,m,n);  %random label init
    niters = 50;
    delta2 = zeros(1,niters);
    wct = 0;
    time2 = delta2;
    
    
    for i=1:niters
        
        %tic;labels = model.fullGibbs_iteration(labels);wct = wct+toc;
        tic;labels = model.collapsedGibbs_iteration(labels);wct = wct+toc;
        %labels = model.mfvb_iteration(labels);
        time2(i) = wct;
        delta2(i) = model.label_log_posterior(labels) - oracle;
        counts = full(sum(labels,2).');
        fprintf('%i: delta = %g, clusters = %i\n',i,delta2(i),sum(counts>0));
    end
    
    
    
    plot(time,delta,time2,delta2,'-*');legend('full','collapsed');
    title(sprintf('D=%i, d=%i, EER=%g',dim,d,EER));
    
    

end


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


function P = sampleP(dim,tame)
    R = rand(dim,dim-1)/tame;
    P = eye(dim) + R*R.';
end


