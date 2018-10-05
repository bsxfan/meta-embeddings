function total = testBackprop_rs(block,X,delta,mask)
%same as testFBblock, but with real step

    if ~iscell(X)
        X = {X};
    end


    dX = cellrndn(X); 
    if exist('mask','var')
       assert(length(mask)==length(X));
       dX = cellmask(dX,mask);
    end
    
    cX1 = cellstep(X,dX,delta);
    cX2 = cellstep(X,dX,-delta);
    DX = cell(size(X)); 
    
    [Y,back] = block(X{:});
    
    
    DY = randn(size(Y));

    [DX{:}] = back(DY);                       %DX = J'*DY
    

    dot1 = celldot(DX,dX);               %dX' * J' * DY
    
    
    cY1 = block(cX1{:});
    cY2 = block(cX2{:});

    [Y2,dY2] = recover(cY1,cY2,delta);   %dY2 = J*dX
    dot2 = DY(:).'*dY2(:);               %DY' * J* DX
    
    
    Y_diff = max(abs(Y(:)-Y2(:))),
    jacobian_diff = abs(dot1-dot2),
    
    
    total = Y_diff + jacobian_diff;
    if total < 1e-6
        fprintf('\ntotal error=%g\n',total);
    else
        fprintf(2,'\ntotal error=%g\n',total);
    end
    

end


function R = cellrndn(X)
    if ~iscell(X)
        R = randn(size(X));
    else
        R = cell(size(X));
        for i=1:numel(X)
            R{i} = cellrndn(X{i});
        end
    end
end


function C = cellstep(X,dX,delta)
    assert(all(size(X)==size(dX)));
    if ~iscell(X)
        C = X + delta*dX;
    else
        C = cell(size(X));
        for i=1:numel(X)
            C{i} = cellstep(X{i},dX{i},delta);
        end
    end
end

function [R,D] = recover(cX1,cX2,delta)
    assert(all(size(cX1)==size(cX2)));
    if ~iscell(cX1)
        R = (cX1+cX2)/2;
        D = (cX1-cX2)/(2*delta);
    else        
        R = cell(size(cX1));
        D = cell(size(cX1));
        for i=1:numel(cX1)
            [R{i},D{i}] = recover(cX1{i},cX2{i}); 
        end
    end
end

function X = cellmask(X,mask)
    if ~iscell(X)
        assert(length(mask)==1);
        X = X*mask;
    else
        for i=1:numel(X)
            X{i} = cellmask(X{i},mask{i});
        end
    end
end


function dot = celldot(X,Y)
    assert(all(size(X)==size(Y)));
    if ~iscell(X)
        dot = X(:).' * Y(:);
    else
        dot = 0;
        for i=1:numel(X)
            dot = dot + celldot(X{i},Y{i});
        end
    end
end

