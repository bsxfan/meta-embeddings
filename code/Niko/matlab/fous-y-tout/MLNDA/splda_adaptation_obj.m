function [y,back] = splda_adaptation_obj(newData,labels,oldF,oldW,params,num_new_Fcols,W_adj_rank,slow)

    if nargin==0
        test_this();
        return;
    end

    if ~exist('slow','var')
        slow = false;
    end

    [dim,Frank] = size(oldF);

    [Fcols,Fscal,Cfac] = unpack(params,dim,Frank,num_new_Fcols,W_adj_rank);
    
    [newF,newW,back1] = adaptSPLDA(Fcols,Fscal,Cfac,oldF,oldW);
    
    [llh,back2] = splda_llh_full(labels,newF,newW,newData,slow,false);

    y = -llh;
    
    
    back = @back_this;
    
    function dparams = back_this(dy)
        dllh = -dy;
        [dnewF,dnewW] = back2(dllh);
        [Fcols,Fscal,Cfac] = back1(dnewF,dnewW);
        dparams = [Fcols(:);Fscal(:);Cfac(:)];
    end
    

end

function [Fcols,Fscal,Cfac] = unpack(params,dim,Frank,num_new_Fcols,W_adj_rank)

    at = 1;
    
    sz = dim*num_new_Fcols;
    Fcols = reshape(params(at:at+sz-1),dim,num_new_Fcols);
    at = at + sz;

    sz = Frank;
    Fscal = reshape(params(at:at+sz-1),1,Frank);
    at = at + sz;
    
    sz = dim*W_adj_rank;
    Cfac = reshape(params(at:at+sz-1),dim,W_adj_rank);
    at = at + sz;
    
    assert(at-1==length(params));
    

end

function test_this()






end