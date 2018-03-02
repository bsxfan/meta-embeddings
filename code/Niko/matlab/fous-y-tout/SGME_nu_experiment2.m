function SGME_nu_experiment2

    zdim = 2;
    rdim = 20;      %required: xdim > zdim
    fscal = 3;      %increase fscal to move speakers apart
    

    F = randn(rdim,zdim)*fscal;
    W = eye(rdim);
    nu = 3;
    
    HTPLDA = create_HTPLDA_extractor(F,nu,W);
    GPLDA = create_GPLDA_extractor(F,W);
    [~,~,dg] = HTPLDA.getPHd();   %dg is same for HTPLDA and GPLDA
    
    n = 5000;
    em = n/10;
    %prior = create_PYCRP(0,[],em,n);   %wide
    prior = create_PYCRP([],0,em,n);    %more concentrated
    
    [R,~,precisions,labels] = sample_HTPLDA_database(nu,F,prior,n);
    niters = 500;
    model_G = SGME_train(R,labels,1000,zdim,niters);
    %[R,~,precisions,labels] = sample_HTPLDA_database(nu(i),F,prior,n);

    model_nu = SGME_train(R,labels,nu,zdim,niters);
    
    
    m = max(labels);
    
    [A,BO] = HTPLDA.extractSGMEs(R,precisions);
    logPsL_Oracle = -SGME_logPsL(A,BO,dg,[],labels,[],prior) / (n*log(m))
        
    [A,BDG] = model_G.extract(R);
    logPsL_DiscG = -SGME_logPsL(A,BDG,model_G.d,[],labels,[],prior) / (n*log(m))

    [A,BDnu] = model_nu.extract(R);
    logPsL_Disc_nu = -SGME_logPsL(A,BDnu,model_nu.d,[],labels,[],prior) / (n*log(m))

    [A,B] = HTPLDA.extractSGMEs(R);
    logPsL_HT = -SGME_logPsL(A,B,dg,[],labels,[],prior) / (n*log(m))
    
    [A,BG] = GPLDA.extractSGMEs(R);
    logPsL_Gauss = -SGME_logPsL(A,B,dg,[],labels,[],prior) / (n*log(m))
    
    close all;
    plot(precisions,B,'.',precisions,BO,'.',precisions,BG,'.',...
         precisions,BDG,'.');
    legend('HT','Oracle','Gauss','Disc:G');
    axis('equal');
    axis('square');
    
    figure;
    plot(precisions,B,'.',precisions,BO,'.',precisions,BG,'.',...
         precisions,BDnu,'.');
    legend('HT','Oracle','Gauss','Disc:\nu');
    axis('equal');
    axis('square');
    
    
    
end