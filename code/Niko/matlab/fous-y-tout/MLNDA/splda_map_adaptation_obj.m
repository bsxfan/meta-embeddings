function [y,back] = splda_map_adaptation_obj(newData,newLabels,...
                                             oldData,oldLabels,...
                                             old_weighting,...
                                             params,Frank,Wfac_numcols,slow)

    if nargin==0
        test_this();
        return;
    end

    
    error('derivatives are not working yet');
    
    if ~exist('slow','var')
        slow = false;
    end

    [dim,~] = size(oldData);

    [F,Wfac] = unpack(params,dim,Frank,Wfac_numcols);
    
    W = Wfac*Wfac.';
    [llh_old,back1] = splda_llh_full(oldLabels,F,W,oldData,slow);
    [llh_new,back2] = splda_llh_full(newLabels,F,W,newData,slow);

    y = -llh_new - old_weighting*llh_old;
    
    
    back = @back_this;
    
    function dparams = back_this(dy)
        % y = -llh_new - old_weighting*llh_old
        dllh_new = -dy;
        dllh_old = -old_weighting*dy;
        
        % [llh_new,back2] = splda_llh_full(newLabels,F,W,newData,slow)
        [dF,dW] = back2(dllh_new);
        
        % [llh_old,back1] = splda_llh_full(oldLabels,F,W,oldData,slow)
        [dF1,dW1] = back1(dllh_old);
        dF = dF + dF1;
        dW = dW + dW1;
        
        % W = Wfac*Wfac.'
        dWfac = 2*dW*Wfac;
        
        dparams = [dF(:);dWfac(:)];
        
        
    end
    

end

function [F,Wfac] = unpack(params,dim,Frank,Wfac_numcols)

    at = 0;
    
    sz = dim*Frank;
    F = reshape(params(at+(1:sz)),dim,Frank);
    at = at + sz;

    sz = dim*Wfac_numcols;
    Wfac = reshape(params(at+(1:sz)),dim,Wfac_numcols);
    at = at + sz;
    
    
    assert( at == length(params) );
    

end

function test_this()

  dim = 20;
  Frank = 5;
  Wfac_numcols = 25;
  
  Nold = 100;
  Kold = 15;
  
  Nnew = 50;
  Knew = 10;

  newData = randn(dim,Nnew);
  newLabels = sparse(randi(Knew,1,Nnew),1:Nnew,true,Knew,Nnew);
  
  oldData = randn(dim,Nold);
  oldLabels = sparse(randi(Kold,1,Nold),1:Nold,true,Kold,Nold);

  old_weighting = 1/4;
  
  F = randn(dim,Frank);
  Wfac = randn(dim,Wfac_numcols);
  params = [F(:);Wfac(:)];
  
  
  f_slow = @(params) splda_map_adaptation_obj(newData,newLabels,...
                                             oldData,oldLabels,...
                                             old_weighting,...
                                             params,Frank,Wfac_numcols,true);
  f_fast = @(params) splda_map_adaptation_obj(newData,newLabels,...
                                             oldData,oldLabels,...
                                             old_weighting,...
                                             params,Frank,Wfac_numcols,false);
  fprintf('test function value equality:\n');
  delta = abs(f_slow(params)-f_fast(params)),
    
  
  fprintf('test slow derivatives:\n');
  testBackprop(f_slow,params);
  
  
  
  
   

end