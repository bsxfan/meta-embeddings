function node = create_SPLDA_node(hlabels,Z,F,W)
% We assume the model has already been diagonalized, so that F'WF is
% diagonal.

    cholW = chol(W);
    [dim,d] = size(F);
    [m,n] = size(hlabels);
    
    E = F'*W*F;  
    E = diag(E);  
    P = F.'*W;  
    
    
    A = [];
    X = [];
    sample();





    node.get = @get;
    node.sample = @sample;
    node.condition_on_child = @condition_on_child;
    node.condition_on_parent = @condition_on_parent;
    node.inferParent = @inferParent;
    node.observe = @observe;

    function val = get()
        val = hlabels;
    end


    function observe(value)
        X = value;
        A = P*X;
    end


    function value = sample()
        X = cholW\randn(dim,n) + F*(Z*hlabels);
        A = P*X;
        value = X;
    end





    function msg = inferParent(p)
        if p==1
            msg = infer_labels();
        elseif p==2
            msg = infer_Z();
        else
            error('bad argument');
        end
    end


    function condition_on_child(arg)
        error('not implemented');
    end



    function condition_on_parent(p,msg)
        if p==1
            condition_on_labels(msg);
        elseif p==2
            condition_on_Z(msg);
        else
            error('bad argument');
        end
    end

    function condition_on_labels(new_hlabels)
        hlabels = new_hlabels;
    end

    function condition_on_Z(new_Z)
        Z = new_Z;
    end


    function LLH = infer_labels()
        ZEZ = E.'*Z.^2;  % 1-by-m
        LLH = bsxfun(@minus,Z.'*A,ZEZ.'/2); %m-by-n
    end

    function msg = infer_Z()
        msg.A = A*hlabels.';  % d-by-m
        counts = sum(hlabels,2);  %m-by-1
        msg.B = E*counts.';   % d-by-m
    end




end