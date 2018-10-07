function [F,W,X,hlabels] = perturb_and_simulate_SPLDA(oldF,oldW,num_new_Fcols,W_adj_rank,nspeakers,recordings_per_speaker)
% Inputs:
%   big: flag to make low or high-dimensional data, each with realistic,
%   single-digit EERs
%
% Outputs:
%   X: i-vectors, dim-by-N
%   hlabels: sparse label matrix, with one hot columns
%   F,W: SPLDA parameters


    [dim,Frank] = size(oldF);
    sigma = sqrt(oldF(:).'*oldF(:)/(dim*Frank));
    Fcols = sigma * randn(dim,num_new_Fcols);
    
    Fscal = 1 + randn(1,Frank)/4;
    
    
    sigma = sqrt(trace(oldW)/dim);
    Cfac = sigma * randn(dim,W_adj_rank);    
    
    
    [F,W] = adaptSPLDA(Fcols,Fscal,Cfac,oldF,oldW);
    
    if nargout > 2
        [rdim,zdim] = size(F);
        nu = inf;
        %Generate synthetic labels
        N = nspeakers*recordings_per_speaker;
        ilabels = repmat(1:nspeakers,recordings_per_speaker,1);
        ilabels = ilabels(:).';  % integer speaker labels
        hlabels = sparse(ilabels,1:N,true,nspeakers,N); %speaker label matrix with one-hot columns

        %and some training data
        Z = randn(zdim,nspeakers);
        X = F*Z*hlabels + sample_HTnoise(nu,rdim,N,W);
    end
    
end


