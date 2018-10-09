function [F,W,X,hlabels] = simulateSPLDA(big,nspeakers,recordings_per_speaker,nu)
% Inputs:
%   big: flag to make low or high-dimensional data, each with realistic,
%   single-digit EERs
%
% Outputs:
%   X: i-vectors, dim-by-N
%   hlabels: sparse label matrix, with one hot columns
%   F,W: SPLDA parameters


    % Assemble model to generate data
    if ~exist('nu','var') || isempty(nu)
        nu = inf;           
    else
        assert(nu>=1)
        %required: nu >= 1, integer, degrees of freedom for heavy-tailed channel noise
    end
    if ~big
        zdim = 2;       %speaker identity variable size 
        rdim = 20;      %i-vector size. required: rdim > zdim
        fscal = 3;      %increase fscal to move speakers apart
    else
        zdim = 100;       %speaker identity variable size 
        rdim = 512;      %i-vector size. required: rdim > zdim
        fscal = 1/20;      %increase fscal to move speakers apart
    end
    
    
    
    F = randn(rdim,zdim)*fscal;
    W = randn(rdim,2*rdim); W = W*W.';W = (rdim/trace(W))*W;
    %model1 = create_HTPLDA_SGME_backend(nu,F,W);  %oracle model
    
    if nargout > 2
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


