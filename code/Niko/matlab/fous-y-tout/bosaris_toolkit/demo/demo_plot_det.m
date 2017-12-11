function demo_plot_det
% Make a DET plot.
% This function is called by 'demo_main'

% generate synthetic scores
tar1 = 2*randn(1,10000)+5;
non1 = randn(1,10000);

tar2 = randn(1,20000)+3;
non2 = randn(1,20000);

% plot DET curves for two systems
plot_title = 'DET plot example';
prior = 0.3;

plot_type = Det_Plot.make_plot_window_from_string('old');
plot_obj = Det_Plot(plot_type,plot_title);

plot_obj.set_system(tar1,non1,'sys1');
plot_obj.plot_steppy_det({'b','LineWidth',2},' ');
plot_obj.plot_DR30_fa('c--','30 false alarms');
plot_obj.plot_DR30_miss('k--','30 misses');
plot_obj.plot_mindcf_point(prior,{'b*','MarkerSize',8},'mindcf');

plot_obj.set_system(tar2,non2,'sys2');
plot_obj.plot_steppy_det({'r','LineWidth',2},' ');
plot_obj.plot_DR30_fa('m--','30 false alarms');
plot_obj.plot_DR30_miss('g--','30 misses');
plot_obj.plot_mindcf_point(prior,{'r*','MarkerSize',8},'mindcf');

plot_obj.display_legend();

fprintf('Look at the figure entitled ''DET plot example'' to see an example of a DET plot.\n');
