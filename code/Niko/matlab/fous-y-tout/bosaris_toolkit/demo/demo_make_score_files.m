function demo_make_score_files(numtar,numnon,scrfilename1,scrfilename2,keyfilename,ndxfilename)
% Create files for scores, keys and indexes and save them in the
% temporary directory so that they can be loaded by the fusion code.

% make some scores
out_data = demo_make_data_for_fusion(numtar,numnon);

scores1 = [out_data.tar1, out_data.non1];
scores2 = [out_data.tar2, out_data.non2];

nummodels = 100;
numsegs = (numtar+numnon)/nummodels;

% make model and segment names
modelset = cellfun(@(x) ['enrol' num2str(x)],num2cell(1:nummodels),'UniformOutput',false);
segset = cellfun(@(x) ['test' num2str(x)],num2cell(1:numsegs),'UniformOutput',false);

% put scores into a matrix
scoremat1 = reshape(scores1,nummodels,numsegs);
scoremat2 = reshape(scores2,nummodels,numsegs);

% create score objects by calling contructors
scr1 = Scores(scoremat1,modelset,segset);
scr2 = Scores(scoremat2,modelset,segset);

% make key
classf = [true(1,numtar),false(1,numnon)];
target_mask = reshape(classf,nummodels,numsegs);
nontarget_mask = ~target_mask;
key = Key(modelset,segset,target_mask,nontarget_mask);

% make index
ndx = key.to_ndx();

% save to file
scr1.save(scrfilename1);
scr2.save(scrfilename2);
key.save(keyfilename);
ndx.save(ndxfilename);
