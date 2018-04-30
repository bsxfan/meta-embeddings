function [Emb,counts,llh,log_prob,i,j] = smart_merge(Emb,counts,llh,prior,llhfun)

    K = length(counts);
    assert(size(Embeddings,2)==K);
    
    LLR = prior.merge_all_pairs(counts);
    for i=1:K-1
        LLR(i,1:i) = -inf;
        jj = i+1:K;
        LLR(i,jj) = llhfun(bsxfun(@plus,Emb(:,i),Emb(:,jj))) -(llh(i)+llh(jj));
    end
    LLR(1,1) = 0;   %don't merge
    
    [mx,sample] = max( LLR(:) + randgumbel(K^2,1) );
    [i,j] = ind2sub(size(LLR),sample);
    

end