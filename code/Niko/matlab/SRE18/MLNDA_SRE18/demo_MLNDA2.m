function demo_MLNDA2()

    % Syntesize SPLDA model and generate data from it
    big = true;
    [F,W,X,hlabels] = simulateSPLDA(big,1000,10);

    
    % Assemble transformation
    rank = 3;
    dim = size(X,1);
    [f,fi,paramsz] = create_nice_Trans2(dim,rank);
    
    % These are the parameters to simulate our test data
    oracle = randn(paramsz,1);
    oracle(1) = sqrt(pi);
    
    % Transform the test data
    T = f(oracle,X);
    
    
    % Initialize to be learnt parameters
    XCovtrace = trace(F*F.'+ inv(W));   % In practice, we have only T, not X. So we
                                        % crudely approximate the covariance of
                                        % X, using the model parameters.
                                  
    TCovtrace = trace(cov(T.',1));
    offset = mean(T,2);
    sigma0 = sqrt(XCovtrace/TCovtrace);
    params0 = [sqrt(sigma0);randn(dim*rank,1)/10;reshape(eye(rank),[],1);offset];
    
    
    
    obj = @(params) MLNDAobj(T,hlabels,F,W,fi,params);    
    obj_oracle = obj(oracle),
    obj_init = obj(params0),
    
    
    maxiters = 100;
    timeout = 20*60;
    [trans,params] = train_ML_trans(F,W,T,hlabels,fi,params0,maxiters,timeout);
    
    % approximately recover X
    % Xhat = trans(T)
    

    obj_oracle = obj(oracle),
    obj_init = obj(params0),
    obj_final = obj(params),
    
end

