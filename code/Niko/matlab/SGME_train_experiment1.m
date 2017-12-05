function SGME_train_experiment1

    zdim = 2;
    rdim = 20;      %required: xdim > zdim
    nu = 3;         %required: nu >= 1, integer, DF
    fscal = 3;      %increase fscal to move speakers apart
    
    F = randn(rdim,zdim)*fscal;

    
    W = eye(rdim);
    
    HTPLDA = create_HTPLDA_extractor(F,nu,W);
    %SGME = HTPLDA.SGME;
    [Pg,Hg,dg] = HTPLDA.getPHd(); 
    
    
    n = 5000;
    em = n/10;
    %prior = create_PYCRP(0,[],em,n);
    prior = create_PYCRP([],0,em,n);
    [R,Z,precisions,labels] = sample_HTPLDA_database(nu,F,prior,n);
    m = max(labels);


    
    
    niters = 500;
    
    model = SGME_train(R,labels,nu,zdim,niters);

    
    fprintf(' ****** Train objectives ******** \n')
    
    [A,B] = model.extract(R);
    discriminative_logPsL = -SGME_logPsL(A,B,model.d,[],labels,[],prior) / (n*log(m))
    calc = create_BXE_calculator(@model.logexpectation,[],labels);
    discriminative_BXE = calc.BXE(A,B) / log(2)
    [tar,non] = calc.get_tar_non(A,B);
    disc_EER = eer(tar,non)
    
    
    [Ag,Bg] = HTPLDA.extractSGMEs(R);
    generative_logPsL = -SGME_logPsL(Ag,Bg,dg,[],labels,[],prior) / (n*log(m))
    calc = create_BXE_calculator(HTPLDA.SGME.log_expectations,[],labels);
    generative_BXE = calc.BXE(Ag,Bg) / log(2)
    [tar,non] = calc.get_tar_non(Ag,Bg);
    gen_EER = eer(tar,non)
    
    

    %generative_objective = model.objective(Pg,Hg,sqrt(dg))
    
%     ntest = 10;
%     emtest = 3;
%     test_prior = create_PYCRP([],0,emtest,ntest);
%     test_labels = test_prior,sample(ntest);
%     Rtest = sample_HTPLDA_database(nu,F,test_labels);
%     
%     [Ag,bg] = extractSGMEs(Rtest);
%     [Ag2,bg2] = SGME_extract(Pg,Hg,nu,Rtest);
    
    fprintf('**** Test objectives ********\n')

    
    %Get fresh test data from same model 
    [R,Z,precisions,labels] = sample_HTPLDA_database(nu,F,prior,n);
    m = max(labels);

    [A,B] = model.extract(R);
    discriminative_logPsL = -SGME_logPsL(A,B,model.d,[],labels,[],prior) / (n*log(m))
    calc = create_BXE_calculator(@model.logexpectation,[],labels);
    discriminative_BXE = calc.BXE(A,B) / log(2)
    [tar,non] = calc.get_tar_non(A,B);
    disc_EER = eer(tar,non)
    
    
    [Ag,Bg] = HTPLDA.extractSGMEs(R);
    generative_logPsL = -SGME_logPsL(Ag,Bg,dg,[],labels,[],prior) / (n*log(m))
    calc = create_BXE_calculator(HTPLDA.SGME.log_expectations,[],labels);
    generative_BXE = calc.BXE(Ag,Bg) / log(2)
    [tar,non] = calc.get_tar_non(Ag,Bg);
    gen_EER = eer(tar,non)
    
    
    


end