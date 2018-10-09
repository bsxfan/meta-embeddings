function [trans,params,obj_final] = train_MAP_trans(F,W,newData,newLabels,...
                                                        oldData,oldLabels,oldWeight,...
                                                        fi,params0,maxiters,timeout,nu)
% Trains a (possibly non-linear) invertible domain adaptation transformation for data from given a
% SPLDA model, using a MP-like criterion. The prior is effected by a
% eighted log-likelihood on th old (source domain) data set.
%
% The recipe is:
%
%     map_params = argmax_(params) log P(newData|f,params,F,W,newLabels)
%                                 + oldWeight * log P(oldData|f,params,F,W,oldLabels)
%
%  where oldData/newData = f(params,X) is an hypothesized transformation of data, X, 
%  generated from an SPLDA model, with parameters F,W and given labels.
%  The likelihood for the transformation parameters, params, can be
%  expanded as follows:
%
%    P(Data|f,params,F,W,Labels) = P(fi(params,Data) | F,W, Labels) / abs(det(J))
%
%  where J is the Jacobian of the forward transformation, f. The inverse of
%  f is dented fi. Note that only fi and the Jacobian determinant are
%  needed during training, while f is needed only when simulating
%  transformed data.
%
%  Inputs:
%    F: SPLDA parameter: dim-by-rank, speaker factor loading matrix
%    W: SPLDA parameter: dim-by-dim, full within speaker precision matrix
%    newData: dim-by-N training data matrix hypothesized to be transformed SPLDA data.
%    newLabels: sparse, logical, K-by-N, label matrix, with one hot columns,
%            for K speakers.
%    oldData,oldLabels: supervised data from the source domain 
%    oldWeight: scalar, less than one, to downweight the effect of the old
%               data log-likelihood
%    fi: inverse of the hypothesized transform f. This will be the
%        transform that `adapts' transformed data so that the SPLDA model
%        with parameters, F,W can be applied for scoring. fi is a function
%        handle: [X,logdetJ,back] = fi(params,Data), where logdetJ is the sum
%        of the log determinants of the Jacobians over all of the data.
%        back is a function handle for backpropagation.
%    params0: initial parameters for the to-be-trained transform
%    maxiters, timeout: control the L-BFGS optimizer.
%    nu: [optional] If supplied, does HT-PLDA (example: nu=2)
%
%   Outputs:
%     trans: function handle, to adapt data before scoring with SPLDA(F,W).
%            Note, the parameters are encapsulated. Use like: X = trans(newData)
%            and then score X in your SPLDA model.
%     params: [optional] the final optized parameters, so that trans = @(Data)
%             fi(params,Data).
%     obj_final: the final value of the optimization objective



    if ~exist('nu','var')
        nu = inf;
    end

    if oldWeight>0
        obj = @(params) MLNDA_MAP_obj(newData,newLabels,oldData,oldLabels,oldWeight,F,W,fi,params,nu);
    else
        obj = @(params) MLNDAobj(newData,newLabels,F,W,fi,params,nu);
    end

    mem = 20;        % L-BFGS memory buffer size
    stpsz0 = 1/100;  % you can play with this to control the initial line 
                     % search step size for the first (often fragile) L-BFGS 
                     % iteration.
    
    [params,obj_final] = L_BFGS(obj,params0,maxiters,timeout,mem,stpsz0);
    
    
    trans = @(Data) fi(params,Data);

end