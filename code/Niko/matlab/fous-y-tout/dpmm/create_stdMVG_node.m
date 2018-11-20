function node = create_stdMVG_node(m,n)


    Z = randn(m,n);
    B = 0;
    Zhat = zeros(m,n);


    node.get = @get;
    node.sample = @sample;
    node.condition_on_child = @condition_on_child;

    function val = get()
        val = Z;
    end

    function val = sample()
        Z = Zhat + randn(m,n)./sqrt(B+1);
        val = Z;
    end


    function condition_on_child(newA,newB)
        B = newB;
        Zhat = newA./(B+1);
    end


    function [logPost,Zmode] = logPosterior_at_mode()
        Zmode = Zhat;
        logPost = sum(log1p(B(:)))/2;
    end

    function logPost = logPosterior_at_zero()
        logPost = (  sum(log1p(B(:))) - sum(Zhat(:)^2./(B(:)+1)) )/2;
    end




end