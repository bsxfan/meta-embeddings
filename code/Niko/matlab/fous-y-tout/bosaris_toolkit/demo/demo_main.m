% this is the main script of the first demo.  Run this script from inside
% Matlab.  Remember to first add the bosaris toolkit to your path
% using addpath(genpath(path_to_bosaris_toolkit))

close all;

fprintf('You need to run this script in Matlab''s graphical mode to see the plots.\n');

% create a Normalized Bayes Error-rate plot
demo_plot_nber;

% create a DET plot
demo_plot_det;

% fuse two systems.

% calculate an effective prior from target prior, Cmiss, and Cfa
prior = effective_prior(0.04,10,1);

% make synthetic scores for fusion demo
numtraintar = 10000;
numtrainnon = 10000;
numtesttar = 5000;
numtestnon = 15000;
train_data = demo_make_data_for_fusion(numtraintar,numtrainnon);
test_data = demo_make_data_for_fusion(numtesttar,numtestnon);

% fuse the two systems
fused_scores = demo_fusion(train_data,test_data,prior);

% split fused scores into target and nontarget scores
test_data.tar_f = fused_scores(1:numtesttar);
test_data.non_f = fused_scores(numtesttar+1:end);

% make a DET plot of the two systems and the fused system.
demo_plot_det_for_fusion(test_data,prior);

% display numerical measures for separate systems and fused system
fprintf('Calculating stats for systems and their fusion.\n');
[actdcf1,mindcf1,prbep1,eer1] = fastEval(test_data.tar1,test_data.non1,prior);
[actdcf2,mindcf2,prbep2,eer2] = fastEval(test_data.tar2,test_data.non2,prior);
[actdcf_f,mindcf_f,prbep_f,eer_f] = fastEval(test_data.tar_f,test_data.non_f,prior);
fprintf('system 1: eer: %5.2f%%; mindcf: %5.2f%%; actdcf: %5.2f%%\n',eer1*100,mindcf1*100,actdcf1*100);
fprintf('system 2: eer: %5.2f%%; mindcf: %5.2f%%; actdcf: %5.2f%%\n',eer2*100,mindcf2*100,actdcf2*100);
fprintf('fused system: eer: %5.2f%%; mindcf: %5.2f%%; actdcf: %5.2f%%\n',eer_f*100,mindcf_f*100,actdcf_f*100);
