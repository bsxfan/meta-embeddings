function [y,back] = MLNDA_MAP_obj(newData,newLabels,oldData,oldLabels,oldWeight,F,W,fi,params)


    if nargin==0
        test_this();
        return;
    end

    [newR,logdetJnew,back1] = fi(params,newData);
    [newllh,back2] = splda_llh(newR,newLabels,F,W);

    [oldR,logdetJold,back3] = fi(params,oldData);
    [oldllh,back4] = splda_llh(oldR,oldLabels,F,W);
    
    
    y = (logdetJnew - newllh) + oldWeight*(logdetJold - oldllh);
    
    back = @back_this;
    
    function dparams = back_this(dy)
        doldR = back4(-oldWeight*dy);
        dnewR = back2(-dy);
        dparams = back3(doldR,oldWeight*dy) + back1(dnewR,dy);
    end


end


function test_this()

    [F,W,oldData,oldLabels] = simulateSPLDA(false,5,2);
    [~,~,newData,newLabels] = simulateSPLDA(false,5,2);
    
    
    rank = 3;
    dim = size(oldData,1);
    [f,fi,paramsz] = create_nice_Trans3(dim,rank);
    params = randn(paramsz,1);
    
    oldWeight = 1/pi;
    
    obj = @(params) MLNDA_MAP_obj(newData,newLabels,oldData,oldLabels,oldWeight,F,W,fi,params);
    
    testBackprop(obj,params);
    
    
end
