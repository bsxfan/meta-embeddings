function [Fcols,Fscal,Cfac] = unpack_SPLDA_adaptation_params(params,dim,Frank,num_new_Fcols,W_adj_rank)

    at = 0;
    
    sz = dim*num_new_Fcols;
    Fcols = reshape(params(at+(1:sz)),dim,num_new_Fcols);
    at = at + sz;

    sz = Frank;
    Fscal = reshape(params(at+(1:sz)),1,Frank);
    at = at + sz;
    
    sz = dim*W_adj_rank;
    Cfac = reshape(params(at+(1:sz)),dim,W_adj_rank);
    at = at + sz;
    
    assert( at == length(params) );
    

end