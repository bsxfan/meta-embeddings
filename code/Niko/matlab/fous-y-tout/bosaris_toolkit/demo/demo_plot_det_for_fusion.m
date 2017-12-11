function demo_plot_det_for_fusion(test_data,prior)
% Display two systems and their fusion on a DET plot.
% This function is called by 'demo_main'

plot_title = 'DET plot for fusion';

plot_type = Det_Plot.make_plot_window_from_string('old');
plot_obj = Det_Plot(plot_type,plot_title);

plot_obj.set_system(test_data.tar1,test_data.non1,'sys1');
plot_obj.plot_steppy_det({'b','LineWidth',2},' ');
plot_obj.plot_mindcf_point(prior,{'b*','MarkerSize',8},'mindcf');

plot_obj.set_system(test_data.tar2,test_data.non2,'sys2');
plot_obj.plot_steppy_det({'r','LineWidth',2},' ');
plot_obj.plot_mindcf_point(prior,{'r*','MarkerSize',8},'mindcf');

plot_obj.set_system(test_data.tar_f,test_data.non_f,'fusion');
plot_obj.plot_steppy_det({'g','LineWidth',2},' ');
plot_obj.plot_mindcf_point(prior,{'g*','MarkerSize',8},'mindcf');

plot_obj.display_legend();

fprintf('Look at the figure entitled ''DET plot for fusion'' to see a DET plot with curves for the two systems and their fusion.\n');
