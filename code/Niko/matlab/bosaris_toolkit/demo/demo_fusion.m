function fused_scores = demo_fusion(train_data,test_data,prior)
% Fuse two systems.
% This function is called by 'demo_main'

% Stack the scores from the two systems.  The two systems must have
% the same number of scores because the scores must be for the same trials.
train_scores = [train_data.tar1, train_data.non1; train_data.tar2, train_data.non2];

% create a label vector to indicate target (+1) and non-target (-1)
% scores (use 0 in this vector to ignore trials).
numtar = length(train_data.tar1);
numnon = length(train_data.non1);
train_labels = [ones(1,numtar), -ones(1,numnon)];

% create and train a function that fuses the scores from the two systems.
quiet = true; % don't display output during fusion training
maxiters = 100; % maximum number of training iterations
obj_func = [];  % use the default objective function: cllr objective
fusion_func = train_linear_fuser(train_scores,train_labels,prior,true,quiet,maxiters,obj_func);

% apply the fusion function to fuse the test scores from the two systems.
test_scores = [test_data.tar1, test_data.non1; test_data.tar2, test_data.non2];
fused_scores = fusion_func(test_scores);

