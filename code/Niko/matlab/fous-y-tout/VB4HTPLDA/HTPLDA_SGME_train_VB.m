function  [backend,obj] = HTPLDA_SGME_train_VB(R,labels,nu,zdim,niters,F,W)
% HTPLDA_SGME_train_VB, implements a mean-field VB algorithm to train the 
% parameters of a heavy-tailed PLDA model. 
%
% The generative model fantasy goes as follows: 
% - The speaker identity variable, z \in R^zdim, is sampled (once) 
% independently for every speaker, from the, standard multivariate normal 
% distribution of dimensionality zdim. 
% - For a speaker represented by z, every new i-vector, r \in R^D, is 
% sampled from the t-distribution, T(F*z, nu, W), where nu is degrees of 
% freedom and W is within-speaker precision.
%
% A clever shortcut is used to compute the approximate, variational hidden 
% variable posteriors in closed form. This speeds up the algorithm. 
% - The shortcut causes the algorithm to be more like EM, than VBEM (which
%   also has an iterative E-step).
% - The shortcut does not give the same solution as the full mean-field 
%   solution, but since the same shortcut is used in runtime scoring it may 
%   give better results than full mean-field VB.
%
%
% Inputs:
%
%   R: D-by-N training data: N i-vectors of dimension D. The model assumes
%      zero mean data. Subtract the mean if this is not the case.
%
%   labels: speaker labels in one of two formats:
%      integer format: N-vector in the range 1..M, where M is the number of
%                      speakers.
%      one-hot format: sparse, logical M-by-N, one hot columns
%                      (This is a large matrix and must be sparse logical.) 
%
%   nu: scalar, positive, degrees of freedom. This parameter is notlearnt 
%       and must be given. Small values give a heavy-tailed model, while 
%       large values give an almost Gaussian model.
%
%   zdim: size of speaker identity variable. It is required that zdim < D,
%         but the accuracy of our shortcut depends on zdim << D. If for
%         example, you have D = 400 or 600, then 100 < zdim < 200 is a good
%         choice.
%
%   niters: number of  training iterations. Look at the VB lower bound plot
%           and the diagnostics printed out during training to choose how
%           many iterations you want to do. You could start with 10.
%   F,W: optional model parameters for initialization. If not given 
%        (recommended) then initialization is done as F = randn(D,zdim)and
%        W = eye(D);
%
% Outputs:
%   
%  backend: an object representing the trained backend, with methods for
%           runtime scoring. See create_HTPLDA_SGME_backend for details
%           of the backend interface.
%
%  obj: the values of the VB lower bound, computed (before the M-step
%       update) for each VB training iteration.


    d = zdim;
    D = size(R,1);
    
    assert(D>d,'zdim must be strictly smaller than data dimensionality');
    
    M = max(labels);
    N = length(labels);
    if ~islogical(labels)
        labels = sparse(labels,1:N,true,M,N);   
    end
    
    
    if ~exist('F','var')
        F = randn(D,d);
        W = eye(D);
    end

    
    obj = zeros(1,niters);
    for i=1:niters
        [F,W,obj(i)] = VB4HTPLDA_iteration(nu,F,W,R,labels);
        fprintf('%i: %g\n',i,obj(i));
    end


    backend = create_HTPLDA_SGME_backend(nu,F,W);

    
    
    
end