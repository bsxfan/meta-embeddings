function MVG_ME_example3

    close all;

    function E = create_me(mu,sigma,R)
        dim = length(mu);
        if exist('R','var') && ~isempty(R)
            B = sigma*eye(dim)+R*R.';
        else
            B = sigma*eye(dim);
        end
        E = create_plain_metaEmb(B*mu,B);
    end



    e1 = create_me([0.3-2;0],1,[2;0]);
    e2 = create_me([0-2;0],1,[0;3]);
    e3 = create_me([2-2;0],3);

    sc = 3;
    e1 = e1.scale(sc);    
    e2 = e2.scale(sc);    
    e3 = e3.scale(sc);    
    
    e12 = e1.add(e2);
    
    figure;hold;
    axis('square');axis('equal');

    prior = create_me([0;0],1);
    plotGaussian(prior,[],'black, dashed','k--');
    
    [mu,C] = e1.get_mu_cov();
    tikz = plotGaussian(mu,C,'blue','b');
    
    [mu,C] = e2.get_mu_cov();
    tikz = plotGaussian(mu,C,'red','r');

    [mu,C] = e3.get_mu_cov();
    tikz = plotGaussian(mu,C,'green','g');

    [mu,C] = e12.get_mu_cov();
    tikz = plotGaussian(mu,C,'magenta','m');

    blue_green = exp(metaEmb_llr(e1,e3)),
    red_green = exp(metaEmb_llr(e2,e3)),
    magenta_green = exp(metaEmb_llr(e12,e3)),
    blue_red = exp(metaEmb_llr(e1,e2)),
    

end