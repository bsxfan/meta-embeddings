function node = create_Label_node(logw,n)


    m = length(logw);
    LLH = 0;
    [~,labels] = max(bsxfun(@minus,logw,log(-log(rand(m,n)))),[],1);
    hlabels = sparse(labels,1:n,true,m,n);
    labels = [];

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
        hlabels = value;
    end


    function counts = inferParent()
        counts = sum(hlabels,2);
    end

    function condition_on_child(new_LLH)
        LLH = new_LLH;
    end

    function condition_on_parent(new_logw)
        logw = new_logw;
    end

    
    function val = sample()
        [~,labels] = max(bsxfun(@plus,logw,LLH - log(-log(rand(m,n)))),[],1);
        hlabels = sparse(labels,1:n,true,m,n);
        val = hlabels;
    end


end