% this is the main script of the second demo.  Run this script from inside
% Matlab.  Remember to first add the bosaris toolkit to your path
% using addpath(genpath(path_to_bosaris_toolkit))

close all;

fprintf('You need to run this script in Matlab''s graphical mode to see the plots.\n');


% create temporary directory to work in
tmp_folder = tempdir;
scoredirlastpart = 'bosdemoscores';
success = mkdir(tmp_folder,scoredirlastpart);
scoredir = fullfile(tmp_folder,scoredirlastpart,'');
if success==0
    fprintf('Could not create directory "%s"\n',scoredir);
else
    fprintf('Created temporary directory "%s" for storing scores.\n',scoredir);
end

% calculate an effective prior from target prior, Cmiss, and Cfa
prior = effective_prior(0.04,10,1);

% create filenames for storing data
dev_scrfilename1 = [scoredir,'/dev_scr1.mat'];
dev_scrfilename2 = [scoredir,'/dev_scr2.mat'];
dev_keyfilename = [scoredir,'/dev_key.mat'];
dev_ndxfilename = [scoredir,'/dev_ndx.mat'];
dev_scorefilenames = {dev_scrfilename1,dev_scrfilename2};
dev_fusedfilename = [scoredir,'/dev_scr_f.mat'];

evl_scrfilename1 = [scoredir,'/evl_scr1.mat'];
evl_scrfilename2 = [scoredir,'/evl_scr2.mat'];
evl_keyfilename = [scoredir,'/evl_key.mat'];
evl_ndxfilename = [scoredir,'/evl_ndx.mat'];
evl_scorefilenames = {evl_scrfilename1,evl_scrfilename2};
evl_fusedfilename = [scoredir,'/evl_scr_f.mat'];

% create data and save to files
fprintf('Creating score, key and ndx files in temporary directory.\n');
numtrntar = 10000;
numtrnnon = 10000;
numtsttar = 5000;
numtstnon = 15000;
demo_make_score_files(numtrntar,numtrnnon,dev_scrfilename1,dev_scrfilename2,dev_keyfilename,dev_ndxfilename);
demo_make_score_files(numtsttar,numtstnon,evl_scrfilename1,evl_scrfilename2,evl_keyfilename,evl_ndxfilename);

% fuse the two systems
fprintf('Fusing two systems and writing dev and eval scores for fused system to temporary directory.\n');
quiet = true; % don't display output during fusion training
maxiters = 100; % maximum number of training iterations
obj_func = [];  % use the default objective function: cllr objective
linear_fusion_dev_eval_from_files(dev_keyfilename,dev_scorefilenames,...
                                  evl_ndxfilename,evl_scorefilenames,...
                                  dev_fusedfilename,evl_fusedfilename,...
                                  prior,maxiters,obj_func,true,quiet);

% read in scores and extract target and non-target scores
fprintf('Reading in fused eval scores from temporary directory and splitting into target and non-target scores.\n');
scr1 = Scores.read(evl_scrfilename1);
scr2 = Scores.read(evl_scrfilename2);
scr_f = Scores.read(evl_fusedfilename);
key = Key.read(evl_keyfilename);
[test_data.tar1,test_data.non1] = scr1.get_tar_non(key);
[test_data.tar2,test_data.non2] = scr2.get_tar_non(key);
[test_data.tar_f,test_data.non_f] = scr_f.get_tar_non(key);

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
