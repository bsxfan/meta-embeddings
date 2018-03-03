function  [HTPLDA,obj] = SGME_train_VB(R,labels,nu,zdim,niters)

    d = zdim;
    D = size(R,1);
    M = max(labels);
    N = length(labels);
    labels = sparse(labels,1:N,true,M,N);   
    
    
    F = randn(D,d);
    W = eye(D);
    obj = zeros(1,niters);
    for i=1:niters
        [F,W,obj(i)] = VB4HTPLDA_iteration(nu,F,W,R,labels);
        fprintf('%i: %g\n',i,obj(i));
    end


    HTPLDA = create_HTPLDA_extractor(F,nu,W);


end