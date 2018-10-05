function [trans,params,obj_final] = train_ML_trans(F,W,T,labels,fi,params0,maxiters,timeout)
% Trains a (possibly non-linear) invertible domain adaptation transformation for data from given a
% SPLDA model, using maximum likelihood.
% The recipe is:
%
%     ml_params = argmax_(params) P(T|f,params,F,W,labels)
%
%  where T = f(params,X) is an hypothesized transformation of data, X, 
%  generated from an SPLDA model, with parameters F,W and given labels.
%  The likelihood for the transformation parameters, params, can be
%  expanded as follows:
%
%    P(T|f,params,F,W,labels) = P(fi(params,T) | F,W, labels) / abs(det(J))
%
%  where J is the Jacobian of the forward transformation, f. The inverse of
%  f is dented fi. Note that only fi and the Jacobian determinant are
%  needed during training, while f is needed only when simulating
%  transformed data.
%
%  Inputs:
%    F: SPLDA parameter: dim-by-rank, speaker factor loading matrix
%    W: SPLDA parameter: dim-by-dim, full within speaker precision matrix
%    T: dim-by-N training data matrix hypothesized to be transformed SPLDA data.
%    labels: sparse, logical, K-by-N, label matrix, with one hot columns,
%            for K speakers.
%    fi: inverse of the hypothesized transform f. This will be the
%        transform that `adapts' transformed data so that the SPLDA model
%        with parameters, F,W can be applied for scoring. fi is a function
%        handle: [X,logdetJ,back] = fi(params,T), where logdetJ is the sum
%        of the log determinants of the Jacobians over all of the data.
%        back is a function handle for backpropagation.
%    params0: initial parameters for the to-be-trained transform
%    maxiters, timeout: control the L-BFGS optimizer.
%
%   Outputs:
%     trans: function handle, to adapt data before scoring with SPLDA(F,W).
%            Note, the parameters are encapsulated. Use like: X = trans(T)
%            and then score X in your SPLDA model.
%     params: [optional] the final optized parameters, so that trans = @(T)
%             fi(params,T).
%     obj_final: the final value of the optimization objective: -log P(T|f,params, ...)





    obj = @(params) MLNDAobj(T,labels,F,W,fi,params);

    mem = 20;        % L-BFGS memory buffer size
    stpsz0 = 1/100;  % you can play with this to control the initial line 
                     % search step size for the first (often fragile) L-BFGS 
                     % iteration.
    
    [params,obj_final] = L_BFGS(obj,params0,maxiters,timeout,mem,stpsz0);
    
    
    trans = @(T) fi(params,T);

end