function SGME_train_experiment_VB

    zdim = 2;
    rdim = 20;      %required: xdim > zdim
    nu = 3;         %required: nu >= 1, integer, DF
    fscal = 3;      %increase fscal to move speakers apart
    
    F = randn(rdim,zdim)*fscal;

    
    W = randn(rdim,rdim+1);W = W*W.'; W = (rdim/trace(W))*W;
    %W = eye(rdim);
    
    HTPLDA = create_HTPLDA_extractor(F,nu,W);
    
    
    n = 10000;
    em = n/10;
    %prior = create_PYCRP(0,[],em,n);
    prior = create_PYCRP([],0,em,n);
    [R,Z,precisions,labels] = sample_HTPLDA_database(nu,F,prior,n,W);
    m = max(labels);

    niters = 30;
    %[backend,obj] = SGME_train_VB(R,labels,nu,zdim,niters);
    [backend,HTPLDA2,obj] = SGME_train_VB(R,labels,nu,zdim,niters);
    close all;
    plot(obj);title('VB lower bound');

    %[nu2,F2,W2] = backend.getParams();
    %HTPLDA3 = create_HTPLDA_extractor(F2,nu2,W2);
    
    
    
    fprintf(' ****** Train objectives ******** \n')
    
    [Ag,Bg,dg] = HTPLDA.extractSGMEs(R);
    logPsL0 = -SGME_logPsL(Ag,Bg,dg,[],labels,[],prior) / (n*log(m))
    calc = create_BXE_calculator(HTPLDA.SGME.log_expectations,[],labels);
    BXE0 = calc.BXE(Ag,Bg) / log(2)
    %[gtar,gnon] = calc.get_tar_non(Ag,Bg);
    %EER0 = eer(gtar,gnon)
    
    [Ag,Bg,dg] = HTPLDA2.extractSGMEs(R);
    logPsL2 = -SGME_logPsL(Ag,Bg,dg,[],labels,[],prior) / (n*log(m))
    calc = create_BXE_calculator(HTPLDA2.SGME.log_expectations,[],labels);
    BXE = calc.BXE(Ag,Bg) / log(2)
    %[gtar,gnon] = calc.get_tar_non(Ag,Bg);
    %EER = eer(gtar,gnon)

    %me = backend.extract(R,true);
    %e3 = backend.log_expectations(me);
    %e2 = HTPLDA2.SGME.log_expectations(me.A,me.b);
    %figure;plot(e2,e3);
    
    %logPsL3 = -SGME_logPsL(me.A,me.b,me.L,[],labels,[],prior) / (n*log(m))
    
    
    
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
    [R,Z,precisions,labels] = sample_HTPLDA_database(nu,F,prior,n,W);
    m = max(labels);

    [Ag,Bg,dg] = HTPLDA.extractSGMEs(R);
    logPsL0 = -SGME_logPsL(Ag,Bg,dg,[],labels,[],prior) / (n*log(m))
    calc = create_BXE_calculator(HTPLDA.SGME.log_expectations,[],labels);
    BXE0 = calc.BXE(Ag,Bg) / log(2)
    %[gtar,gnon] = calc.get_tar_non(Ag,Bg);
    %EER0 = eer(gtar,gnon)
    
    [Ag,Bg,dg] = HTPLDA2.extractSGMEs(R);
    logPsL = -SGME_logPsL(Ag,Bg,dg,[],labels,[],prior) / (n*log(m))
    calc = create_BXE_calculator(HTPLDA2.SGME.log_expectations,[],labels);
    BXE = calc.BXE(Ag,Bg) / log(2)
    %[gtar,gnon] = calc.get_tar_non(Ag,Bg);
    %EER = eer(gtar,gnon)
    
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