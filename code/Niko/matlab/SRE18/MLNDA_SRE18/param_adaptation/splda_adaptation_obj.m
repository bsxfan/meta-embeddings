function [y,back] = splda_adaptation_obj(newData,labels,oldF,oldW,params,num_new_Fcols,W_adj_rank,slow)

    if nargin==0
        test_this();
        return;
    end

    if ~exist('slow','var')
        slow = false;
    end

    [dim,Frank] = size(oldF);

    [Fcols,Fscal,Cfac] = unpack_SPLDA_adaptation_params(params,dim,Frank,num_new_Fcols,W_adj_rank);
    
    [newF,newW,back1] = adaptSPLDA(Fcols,Fscal,Cfac,oldF,oldW);
    
    [llh,back2] = splda_llh_full(labels,newF,newW,newData,slow);

    y = -llh;
    
    
    back = @back_this;
    
    function dparams = back_this(dy)
        dllh = -dy;
        [dnewF,dnewW] = back2(dllh);
        [Fcols,Fscal,Cfac] = back1(dnewF,dnewW);
        dparams = [Fcols(:);Fscal(:);Cfac(:)];
    end
    

end


function test_this()

  dim = 20;
  Frank = 5;
  num_new_Fcols = 2;
  W_adj_rank = 3;
  n = 100;
  K = 15;
  

  newData = randn(dim,n);
  labels = sparse(randi(K,1,n),1:n,true,K,n);
  
  oldF = randn(dim,Frank);
  oldW = randn(dim,dim+1);oldW = oldW * oldW.';
  Fcols = randn(dim,num_new_Fcols);
  Cfac = randn(dim,W_adj_rank);
  Fscal = randn(1,Frank);
  params = [Fcols(:);Fscal(:);Cfac(:)];
  
  
  f_slow = @(params) splda_adaptation_obj(newData,labels,oldF,oldW,params,num_new_Fcols,W_adj_rank,true);
  f_fast = @(params) splda_adaptation_obj(newData,labels,oldF,oldW,params,num_new_Fcols,W_adj_rank,false);
  
  fprintf('test function value equality:\n');
  delta = abs(f_slow(params)-f_fast(params)),
    
    
  fprintf('test slow derivatives:\n');
  testBackprop(f_slow,params);
  
  
    [~,back] = f_slow(params);
    dparams = back(pi);
    
    [~,back] = f_fast(params);
    dparams = back(pi);
    
    
    fprintf('compare fast and slow derivatives:\n');
    delta = max(abs(dparams-dparams)),
  
   

end