function out_data = demo_make_data_for_fusion(numtar,numnon)
% Make scores for two systems so that they can be fused.

out_data.tar1 = 2*randn(1,numtar)+5;
out_data.non1 = randn(1,numnon);
out_data.tar2 = randn(1,numtar)+3;
out_data.non2 = randn(1,numnon);
