function [R,Z,precisions,labels] = sample_HTPLDA_database(nu,F,labels,n,W)

    if isstruct(labels)  %labels is prior
        prior = labels;
        labels = prior.sample(n);
    else
        assert(nargin==3,'if labels are given, do not also give n');
        n = length(labels);  %no of recordings
    end

    
    
    
    
    [D,dim] = size(F);
    ns = max(labels);    %no of speakers
    subsets = sparse(labels2blocks(labels));
    Z = randn(dim,ns);   %speaker variables

    
    if ~exist('W','var') || isempty(W)
        [X,precisions] = sample_HTnoise(nu,D,n);
    else
        [X,precisions] = sample_HTnoise(nu,D,n,W);
    end
    
    
    
    
    R = F*(Z*subsets.') + X;

end