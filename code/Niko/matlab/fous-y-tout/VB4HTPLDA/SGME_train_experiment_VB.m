function SGME_train_experiment_VB

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

    niters = 10;
    [HTPLDA2,obj] = SGME_train_VB(R,labels,nu,zdim,niters);
    close all;
    plot(obj);title('VB lower bound');
    
    
    
    fprintf(' ****** Train objectives ******** \n')
    
    [Ag,Bg] = HTPLDA.extractSGMEs(R);
    logPsL0 = -SGME_logPsL(Ag,Bg,dg,[],labels,[],prior) / (n*log(m))
    calc = create_BXE_calculator(HTPLDA.SGME.log_expectations,[],labels);
    BXE0 = calc.BXE(Ag,Bg) / log(2)
    [gtar,gnon] = calc.get_tar_non(Ag,Bg);
    EER0 = eer(gtar,gnon)
    
    [Ag,Bg] = HTPLDA2.extractSGMEs(R);
    logPsL = -SGME_logPsL(Ag,Bg,dg,[],labels,[],prior) / (n*log(m))
    calc = create_BXE_calculator(HTPLDA2.SGME.log_expectations,[],labels);
    BXE = calc.BXE(Ag,Bg) / log(2)
    [gtar,gnon] = calc.get_tar_non(Ag,Bg);
    EER = eer(gtar,gnon)
    
%     close all;
%     
%     plot_type = Det_Plot.make_plot_window_from_string('old');
%     plot_obj = Det_Plot(plot_type,'Disc vs Gen');
%     plot_obj.set_system(dtar,dnon,'disc');
%     plot_obj.plot_rocch_det({'b'},'train');    
%     plot_obj.set_system(gtar,gnon,'gen');
%     plot_obj.plot_rocch_det({'g'},'train');    
%     
%     calplot = Norm_DCF_Plot([-8,4,0.0,1.4],'Train');    
%     calplot.set_system(gtar,gnon,'gen_{min}');
%     plot_dcf_curve_min(calplot,{'g--'},'train');
%     calplot.set_system(gtar,gnon,'gen');
%     plot_dcf_curve_act(calplot,{'g'},'train');
%     
%     calplot.set_system(dtar,dnon,'disc_{min}');
%     plot_dcf_curve_min(calplot,{'r--'},'train');
%     calplot.set_system(dtar,dnon,'disc');
%     plot_dcf_curve_act(calplot,{'r'},'train');
% 
%     calplot.set_system(dtar,dnon,'');
%     plot_DR30_fa(calplot,{'k<','MarkerFaceColor','k','MarkerSize',8},'30 false alarms');
%     plot_DR30_miss(calplot,{'k>','MarkerFaceColor','k','MarkerSize',8},'30 misses');
%     
%     display_legend(calplot);
%     
    
    
    
    fprintf('**** Test objectives ********\n')

    
    %Get fresh test data from same model 
    [R,Z,precisions,labels] = sample_HTPLDA_database(nu,F,prior,n);
    m = max(labels);

    [Ag,Bg] = HTPLDA.extractSGMEs(R);
    logPsL0 = -SGME_logPsL(Ag,Bg,dg,[],labels,[],prior) / (n*log(m))
    calc = create_BXE_calculator(HTPLDA.SGME.log_expectations,[],labels);
    BXE0 = calc.BXE(Ag,Bg) / log(2)
    [gtar,gnon] = calc.get_tar_non(Ag,Bg);
    EER0 = eer(gtar,gnon)
    
    [Ag,Bg] = HTPLDA2.extractSGMEs(R);
    logPsL = -SGME_logPsL(Ag,Bg,dg,[],labels,[],prior) / (n*log(m))
    calc = create_BXE_calculator(HTPLDA2.SGME.log_expectations,[],labels);
    BXE = calc.BXE(Ag,Bg) / log(2)
    [gtar,gnon] = calc.get_tar_non(Ag,Bg);
    EER = eer(gtar,gnon)
    
%     plot_obj.set_system(dtar,dnon,'disc');
%     plot_obj.plot_rocch_det({'r--'},'test');    
%     plot_obj.set_system(gtar,gnon,'gen');
%     plot_obj.plot_rocch_det({'m--'},'test');    
%     plot_obj.display_legend();    
%     
% 
%     calplot = Norm_DCF_Plot([-8,4,0.0,1.4],'Test');    
%     calplot.set_system(gtar,gnon,'gen_{min}');
%     plot_dcf_curve_min(calplot,{'g--'},'test');
%     calplot.set_system(gtar,gnon,'gen');
%     plot_dcf_curve_act(calplot,{'g'},'test');
%     
%     calplot.set_system(dtar,dnon,'disc_{min}');
%     plot_dcf_curve_min(calplot,{'r--'},'test');
%     calplot.set_system(dtar,dnon,'disc');
%     plot_dcf_curve_act(calplot,{'r'},'test');
% 
%     calplot.set_system(dtar,dnon,'');
%     plot_DR30_fa(calplot,{'k<','MarkerFaceColor','k','MarkerSize',8},'30 false alarms');
%     plot_DR30_miss(calplot,{'k>','MarkerFaceColor','k','MarkerSize',8},'30 misses');
%     display_legend(calplot);
    
    
end