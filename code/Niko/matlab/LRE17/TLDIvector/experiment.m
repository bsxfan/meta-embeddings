function experiment

    dim = 400;
    N = 100;
    fdim = 10;
    K = 10;
    L = 5;
    
    T = randn(K*fdim,dim);
    M = randn(dim,L);
    nu = 2;
    minDur = 300;
    maxDur = 3000;
    W = eye(dim);
    alpha = [];

    fprintf('generating data:\n');
    [F,Z,labels,X,lambda] = rand_ivector(M,nu,W,alpha,K,T,minDur,maxDur,N);
    [labels,jj] = find(labels);
    
    fprintf('precomputing TT:\n');
    TT = precomputeTT(T,fdim,dim,K);
    B = TT*Z;
    A = T.'*F;
    
    R = chol(W);
    RA = (R.')\A;
    RM = R*M;
    
    fprintf('estimating lambdas:\n');
    lambda_star = zeros(1,N);
    for i=1:N
        ell = labels(i);
        C = create_diagonalized_C(reshape(B(:,i),dim,dim),R,RM,RA(:,i));
        lambda_star(i) = exp(C.lambda_by_root(nu,0,ell));
        fprintf('%i: %g, %g\n',i,lambda(i),lambda_star(i));
    end
    
    close all;
    plot(lambda,lambda_star(:),'.');



end