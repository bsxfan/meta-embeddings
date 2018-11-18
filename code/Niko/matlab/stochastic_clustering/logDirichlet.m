function logP = logDirichlet(p,alpha)
    p = p(:);
    if isscalar(alpha)
        alpha = alpha*ones(size(p));
    else
        alpha = alpha(:);
        assert(length(alpha)==length(p));
    end
    logP = sum(log(p).*(alpha-1),1) + gammaln(sum(alpha,1)) - sum(gammaln(alpha),1);
end