function SGME_demo1

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
    timeout = 5*60;
    model1 = SGME_train(R,labels,zdim,[],niters,timeout);
    model2 = SGME_train(R,labels,zdim,nu,niters,timeout);

    doplots = false;
    
    
    fprintf(' ****** Train objectives ******** \n')
    
    [bxe,tar_non] = create_SGME_BXE_calculator(labels);

    [w_extr,w_exp] = model1.getParams();
    E = model1.extract(w_extr,R);
    dBXE1 = bxe(E,w_exp)
    [tar,non] = tar_non(E,w_exp);
    dEER1 = eer(tar,non)
    
    [w_extr,w_exp] = model2.getParams();
    E = model2.extract(w_extr,R);
    dBXE2 = bxe(E,w_exp)
    [tar,non] = tar_non(E,w_exp);
    dEER2 = eer(tar,non)


    fprintf(' ****** Test objectives ******** \n')
    [R,Z,precisions,labels] = sample_HTPLDA_database(nu,F,prior,n);
    
    [bxe,tar_non] = create_SGME_BXE_calculator(labels);

    [w_extr,w_exp] = model1.getParams();
    E = model1.extract(w_extr,R);
    dBXE1 = bxe(E,w_exp)
    [tar,non] = tar_non(E,w_exp);
    dEER1 = eer(tar,non)
    
    [w_extr,w_exp] = model2.getParams();
    E = model2.extract(w_extr,R);
    dBXE2 = bxe(E,w_exp)
    [tar,non] = tar_non(E,w_exp);
    dEER2 = eer(tar,non)
    
    
    
    return;
    
    discriminative_logPsL = -SGME_logPsL(A,B,model.d,[],labels,[],prior) / (n*log(m))
    calc = create_BXE_calculator(@model.logexpectation,[],labels);
    discriminative_BXE = calc.BXE(A,B) / log(2)
    
    
    [Ag,Bg] = HTPLDA.extractSGMEs(R);
    generative_logPsL = -SGME_logPsL(Ag,Bg,dg,[],labels,[],prior) / (n*log(m))
    calc = create_BXE_calculator(HTPLDA.SGME.log_expectations,[],labels);
    generative_BXE = calc.BXE(Ag,Bg) / log(2)
    [gtar,gnon] = calc.get_tar_non(Ag,Bg);
    gen_EER = eer(gtar,gnon)
    
    
    close all;
    
    
    if doplots
    
        plot_type = Det_Plot.make_plot_window_from_string('old');
        plot_obj = Det_Plot(plot_type,'Disc vs Gen');
        plot_obj.set_system(dtar,dnon,'disc');
        plot_obj.plot_rocch_det({'b'},'train');    
        plot_obj.set_system(gtar,gnon,'gen');
        plot_obj.plot_rocch_det({'g'},'train');    

        calplot = Norm_DCF_Plot([-8,4,0.0,1.4],'Train');    
        calplot.set_system(gtar,gnon,'gen_{min}');
        plot_dcf_curve_min(calplot,{'g--'},'train');
        calplot.set_system(gtar,gnon,'gen');
        plot_dcf_curve_act(calplot,{'g'},'train');

        calplot.set_system(dtar,dnon,'disc_{min}');
        plot_dcf_curve_min(calplot,{'r--'},'train');
        calplot.set_system(dtar,dnon,'gen');
        plot_dcf_curve_act(calplot,{'r'},'train');

        calplot.set_system(dtar,dnon,'');
        plot_DR30_fa(calplot,{'k<','MarkerFaceColor','k','MarkerSize',8},'30 false alarms');
        plot_DR30_miss(calplot,{'k>','MarkerFaceColor','k','MarkerSize',8},'30 misses');

        display_legend(calplot);
    end
    
    
    
    
    fprintf('**** Test objectives ********\n')

    
    %Get fresh test data from same model 
    [R,Z,precisions,labels] = sample_HTPLDA_database(nu,F,prior,n);
    m = max(labels);

    [A,B] = model.extract(R);
    discriminative_logPsL = -SGME_logPsL(A,B,model.d,[],labels,[],prior) / (n*log(m))
    calc = create_BXE_calculator(@model.logexpectation,[],labels);
    discriminative_BXE = calc.BXE(A,B) / log(2)
    [dtar,dnon] = calc.get_tar_non(A,B);
    disc_EER = eer(dtar,dnon)
    
    
    [Ag,Bg] = HTPLDA.extractSGMEs(R);
    generative_logPsL = -SGME_logPsL(Ag,Bg,dg,[],labels,[],prior) / (n*log(m))
    calc = create_BXE_calculator(HTPLDA.SGME.log_expectations,[],labels);
    generative_BXE = calc.BXE(Ag,Bg) / log(2)
    [gtar,gnon] = calc.get_tar_non(Ag,Bg);
    gen_EER = eer(gtar,gnon)
    
    
    if doplots
        plot_obj.set_system(dtar,dnon,'disc');
        plot_obj.plot_rocch_det({'r--'},'test');    
        plot_obj.set_system(gtar,gnon,'gen');
        plot_obj.plot_rocch_det({'m--'},'test');    
        plot_obj.display_legend();    


        calplot = Norm_DCF_Plot([-8,4,0.0,1.4],'Test');    
        calplot.set_system(gtar,gnon,'gen_{min}');
        plot_dcf_curve_min(calplot,{'g--'},'test');
        calplot.set_system(gtar,gnon,'gen');
        plot_dcf_curve_act(calplot,{'g'},'test');

        calplot.set_system(dtar,dnon,'disc_{min}');
        plot_dcf_curve_min(calplot,{'r--'},'test');
        calplot.set_system(dtar,dnon,'gen');
        plot_dcf_curve_act(calplot,{'r'},'test');

        calplot.set_system(dtar,dnon,'');
        plot_DR30_fa(calplot,{'k<','MarkerFaceColor','k','MarkerSize',8},'30 false alarms');
        plot_DR30_miss(calplot,{'k>','MarkerFaceColor','k','MarkerSize',8},'30 misses');
        display_legend(calplot);
    end
    
    
end