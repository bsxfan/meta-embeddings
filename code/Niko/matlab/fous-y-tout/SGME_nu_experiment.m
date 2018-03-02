function SGME_nu_experiment

    zdim = 2;
    rdim = 20;      %required: xdim > zdim
    fscal = 3;      %increase fscal to move speakers apart
    

    F = randn(rdim,zdim)*fscal;
    W = eye(rdim);
    GPLDA = create_HTPLDA_extractor(F,1000,W);
    [~,~,dg] = GPLDA.getPHd(); 
    
    n = 5000;
    em = n/10;
    %prior = create_PYCRP(0,[],em,n);
    prior = create_PYCRP([],0,em,n);
    
    nu = 1:5;
    scal = exp(log(100)*(-10:10)/10); 
    logPsL = zeros(length(nu),length(scal));
    %logPsL_G = zeros(length(nu),1);
    logPsL_Oracle = zeros(length(nu),1);
    logPsL_D = zeros(length(nu),1);
    for i=1:length(nu)
        fprintf('\n\n sampling data, nu = %i\n',nu(i));
        [R,~,~,labels] = sample_HTPLDA_database(nu(i),F,prior,n);
        niters = 500;
        model = SGME_train(R,labels,1000,zdim,niters);
        [R,~,precisions,labels] = sample_HTPLDA_database(nu(i),F,prior,n);

        
        m = max(labels);
        [A,B] = GPLDA.extractSGMEs(R,precisions);
        logPsL_Oracle(i) = -SGME_logPsL(A,B,dg,[],labels,[],prior) / (n*log(m));
        [A,B] = model.extract(R);
        logPsL_D(i) = -SGME_logPsL(A,B,model.d,[],labels,[],prior) / (n*log(m));
        for j=1:length(scal)
            HTPLDA = create_HTPLDA_extractor(F,nu(i)*scal(j),W);
            [A,B] = HTPLDA.extractSGMEs(R);
            logPsL(i,j) = -SGME_logPsL(A,B,dg,[],labels,[],prior) / (n*log(m));
            fprintf('  scal = %g, PsL = %g\n',scal(j),logPsL(i,j));
        end
    end
    
    
    
    close all;
    delta = bsxfun(@minus,logPsL,logPsL_Oracle);
    semilogx(scal,delta');
    legend('\nu = 1','\nu = 2','\nu = 3','\nu = 4','\nu = 5');
    grid;

    figure();
    delta = bsxfun(@minus,logPsL,logPsL_D);
    semilogx(scal,delta');
    legend('\nu = 1','\nu = 2','\nu = 3','\nu = 4','\nu = 5');
    grid;
    
    M = [logPsL_Oracle, logPsL_D, logPsL(:,11)]
    
    
    
end