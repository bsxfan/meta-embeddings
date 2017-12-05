function demo_plot_nber
% Make a Normalized Bayes Error-rate plot.
% this function is called by 'demo_main'

% make synthetic scores to plot
tar = 2*randn(1,10000)+3;
non = randn(1,100000);
LRtar = -0.5*(tar-3).^2/(2^2) + 0.5*tar.^2 - 0.5*log(2);
LRnon = -0.5*(non-3).^2/(2^2) + 0.5*non.^2 - 0.5*log(2);


% make the plot
plot_obj = Norm_DCF_Plot([-10,10,0.2,1.4],'Synthetic LR');

plot_obj.set_system(0*LRtar,0*LRnon,'log LR = 0');
plot_dcf_curve_act(plot_obj,{'k','LineWidth',2},' ');

plot_obj.set_system(LRtar,LRnon,'E_{min}');
plot_dcf_curve_min(plot_obj,{'k--','LineWidth',2},' ');

plot_obj.set_system(LRtar,LRnon,'true log LR');
plot_dcf_curve_act(plot_obj,{'g','LineWidth',1},' ');

plot_obj.set_system(2*LRtar,2*LRnon,'2 \times log LR');
plot_dcf_curve_act(plot_obj,{'r','LineWidth',1},' ');

plot_obj.set_system(0.5*LRtar,0.5*LRnon,'0.5 \times log LR');
plot_dcf_curve_act(plot_obj,{'r--','LineWidth',1},' ');

plot_obj.set_system(2+LRtar,2+LRnon,'log LR + 2');
plot_dcf_curve_act(plot_obj,{'m','LineWidth',1},' ');

plot_obj.set_system(-2+LRtar,-2+LRnon,'log LR - 2');
plot_dcf_curve_act(plot_obj,{'m--','LineWidth',1},' ');

plot_obj.set_system(LRtar,LRnon,'');
plot_DR30_fa(plot_obj,{'rV','MarkerFaceColor','r','MarkerSize',8},'30 false alarms');
plot_DR30_miss(plot_obj,{'gV','MarkerFaceColor','g','MarkerSize',8},'30 misses');

display_legend(plot_obj);

xlabel('logit \pi');
ylabel('normalized Bayes error-rate');

fprintf('Look at the figure entitled ''Synthetic LR'' to see an example of a normalized Bayes error-rate plot.\n');
