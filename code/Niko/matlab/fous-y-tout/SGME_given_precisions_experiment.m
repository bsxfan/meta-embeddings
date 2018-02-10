function SGME_given_precisions_experiment

    zdim = 2;
    rdim = 20;      %required: xdim > zdim
    nu = 5;         %required: nu >= 1, integer, DF
    fscal = 3;      %increase fscal to move speakers apart
    
    F = randn(rdim,zdim)*fscal;

    
    W = eye(rdim);
    
    HTPLDA = create_HTPLDA_extractor(F,nu,W);
    GPLDA = create_HTPLDA_extractor(F,1000,W);

    
    [~,~,dg] = HTPLDA.getPHd(); 
    
    
    n = 5000;
    em = n/10;
    %prior = create_PYCRP(0,[],em,n);
    prior = create_PYCRP([],0,em,n);
    [R,Z,precisions,labels] = sample_HTPLDA_database(nu,F,prior,n);
    m = max(labels);


    
    

    
    fprintf(' ****** Test objectives ******** \n')
    
    [Ag,Bg] = HTPLDA.extractSGMEs(R,precisions);
    generative_logPsL = -SGME_logPsL(Ag,Bg,dg,[],labels,[],prior) / (n*log(m))
    calc = create_BXE_calculator(HTPLDA.SGME.log_expectations,[],labels);
    generative_BXE = calc.BXE(Ag,Bg) / log(2)
    [gtar,gnon] = calc.get_tar_non(Ag,Bg);
    gen_EER = eer(gtar,gnon)
    
    [Ag,Bg] = HTPLDA.extractSGMEs(R);
    generative_logPsL = -SGME_logPsL(Ag,Bg,dg,[],labels,[],prior) / (n*log(m))
    calc = create_BXE_calculator(HTPLDA.SGME.log_expectations,[],labels);
    generative_BXE = calc.BXE(Ag,Bg) / log(2)
    [gtar,gnon] = calc.get_tar_non(Ag,Bg);
    gen_EER = eer(gtar,gnon)

    [AG,BG] = GPLDA.extractSGMEs(R);
    G_logPsL = -SGME_logPsL(AG,BG,dg,[],labels,[],prior) / (n*log(m))
    calc = create_BXE_calculator(GPLDA.SGME.log_expectations,[],labels);
    G_BXE = calc.BXE(AG,BG) / log(2)
    [Gtar,Gnon] = calc.get_tar_non(AG,BG);
    G_EER = eer(Gtar,Gnon)
    
    return;
    
    close all;
    
    plot_type = Det_Plot.make_plot_window_from_string('old');
    plot_obj = Det_Plot(plot_type,'G vs HT');
    plot_obj.set_system(Gtar,Gnon,'G');
    plot_obj.plot_rocch_det({'b'},'test');    
    plot_obj.set_system(gtar,gnon,'HT');
    plot_obj.plot_rocch_det({'g'},'test');    
    
    calplot = Norm_DCF_Plot([-8,4,0.0,1.4],'Test');    
    calplot.set_system(gtar,gnon,'HT_{min}');
    plot_dcf_curve_min(calplot,{'g--'},'test');
    calplot.set_system(gtar,gnon,'HT');
    plot_dcf_curve_act(calplot,{'g'},'test');
    
    calplot.set_system(Gtar,Gnon,'G_{min}');
    plot_dcf_curve_min(calplot,{'r--'},'test');
    calplot.set_system(Gtar,Gnon,'G');
    plot_dcf_curve_act(calplot,{'r'},'test');

    calplot.set_system(Gtar,Gnon,'');
    plot_DR30_fa(calplot,{'k<','MarkerFaceColor','k','MarkerSize',8},'30 false alarms');
    plot_DR30_miss(calplot,{'k>','MarkerFaceColor','k','MarkerSize',8},'30 misses');
    
    display_legend(calplot);
    
    
    
    
end