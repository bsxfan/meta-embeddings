function experiment

    dim = 400;
    N = 100;
    fdim = 40;
    K = 1000;
    L = 5;
    
    T = randn(K*fdim,dim);
    M = randn(dim,L);
    nu = 0.05;
    minDur = 300;
    maxDur = 3000;
    %minDur = 0;
    %maxDur = 30;
    W = eye(dim);
    alpha = [];

    fprintf('generating data:\n');
    [F,Z,labels,X,lambda,dur] = rand_ivector(M,nu,W,alpha,K,T,minDur,maxDur,N);
    [labels,jj] = find(labels);
    
    fprintf('precomputing TT:\n');
    TT = precomputeTT(T,fdim,dim,K);
    B = TT*Z;
    A = T.'*F;
    
    R = chol(W);
    RA = (R.')\A;
    RM = R*M;
    
    fprintf('estimating lambdas:\n');
    lambda_root = zeros(1,N);
    lambda_fp = zeros(1,N);
    for i=1:N
        ell = labels(i);
        C = create_diagonalized_C(reshape(B(:,i),dim,dim),R,RM,RA(:,i));
        lambda_root(i) = C.lambda_by_root(nu,0,ell);
        lambda_fp(i) = C.lambda_by_fixed_point(nu,0,ell,1);
        fprintf('%i: (%i) %g, %g, %g\n',i,dur(i),log(lambda(i)),lambda_root(i),lambda_fp(i));
    end
    
    close all;
    subplot(1,3,1);plot(log(lambda),lambda_root,'r.');grid;axis('square');axis('equal');
    subplot(1,3,2);plot(log(lambda),lambda_fp,'g.');grid;axis('square');axis('equal');
    subplot(1,3,3);plot(lambda_root,lambda_fp,'k.');grid;axis('square');axis('equal');



end