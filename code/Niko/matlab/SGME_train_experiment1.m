function SGME_train_experiment1

    zdim = 2;
    rdim = 20;      %required: xdim > zdim
    nu = 3;         %required: nu >= 1, integer, DF
    fscal = 3;      %increase fscal to move speakers apart
    
    F = randn(rdim,zdim)*fscal;

    
    %HTPLDA = create_HTPLDA_extractor(F,nu);
    %SGME = HTPLDA.SGME;
    
    
    n = 1000;
    em = 100;
    %prior = create_PYCRP(0,[],em,n);
    prior = create_PYCRP([],0,em,n);
    [R,Z,precisions,labels] = sample_HTPLDA_database(nu,F,prior,n);


    niters = 500;
    
    model = SGME_train(R,labels,nu,zdim,niters);




end