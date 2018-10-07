function [newF,newW] = train_ML_adapted_SPLDA(oldF,oldW,newData,newLabels,...
                                              num_new_Fcols,W_adj_rank, ...
                                              maxiters,timeout)


    obj = @(params) splda_adaptation_obj(newData,newLabels,oldF,oldW,params,num_new_Fcols,W_adj_rank);

    
    %initialize adaptation parameters
    [dim,Frank] = size(oldF);
    sigma = sqrt(oldF(:).'*oldF(:)/(dim*Frank));
    Fcols0 = (sigma/10) * randn(dim,num_new_Fcols);
    
    Fscal0 = ones(1,Frank);
    
    
    sigma = sqrt(trace(oldW)/dim);
    Cfac0 = (sigma/10) * randn(dim,W_adj_rank);
    
    params0 = [Fcols0(:);Fscal0(:);Cfac0(:)];
    
    
    mem = 20;        % L-BFGS memory buffer size
    stpsz0 = 1/100;  % you can play with this to control the initial line 
                     % search step size for the first (often fragile) L-BFGS 
                     % iteration.
    
    [params,obj_final] = L_BFGS(obj,params0,maxiters,timeout,mem,stpsz0);
    
    [Fcols,Fscal,Cfac] = unpack_SPLDA_adaptation_params(params,dim,Frank,num_new_Fcols,W_adj_rank);
    [newF,newW] = adaptSPLDA(Fcols,Fscal,Cfac,oldF,oldW);

end