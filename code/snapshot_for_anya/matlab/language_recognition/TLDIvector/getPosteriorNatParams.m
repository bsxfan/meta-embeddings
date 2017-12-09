function [A,B,k,n,TT] = getPosteriorNatParams(stats_or_ivectors,Z,T,TT)
% Computes the data-dependent terms of the natural parameters of the
% i-vector posterior. The data independent terms are formed by the prior.
%
% Inputs:
%   stats_or_ivectors:  can be either F, or ivectors
%       F: dm-by-n first-order stats
%       ivectors: k-by-n, classical i-vector point-estimates
%   Z: m-by-n zero order stats
%   T: dm-by-k factor loading matrix
%
% Outputs:
%   A: k-by-n, data-dependent term of posterior natural mean 
%   B: (k^2)-by-n, vectorized data dependent term of posterior precision matrices 

    [m,n] = size(Z);  %m components, n segments
    [dm,k] = size(T);  %k is ivector dimension
    d = dm/m; %feature dimension
    
    %precompute big 2nd-order stuff
    if ~exist('TT','var') || isempty(TT)
        TT = precomputeTT(T,d,k,m);
    end

    B = TT*Z; % k^2-by-n, data-dependent posterior precision terms

    [sz,n2] = size(stats_or_ivectors);  
    assert(n2==n);
    if sz==dm
        F = stats_or_ivectors;
        A = T.'*F;  % k-by-n projected first-order stats
    elseif sz==k
        A = stats_or_ivectors;  %here ivectors
        for t=1:n
            Bt = reshape(B(:,t),k,k);
            A(:,t) = A(:,t) + Bt*stats_or_ivectors(:,t);
        end
    else
        error('stats_or_ivectors has invalid size');
    end


end