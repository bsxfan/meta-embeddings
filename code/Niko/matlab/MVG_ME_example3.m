function MVG_ME_example3
% Plots Gaussians and computes LRs for example in document at:
% Practical meta-embeddings: Multivariate gaussian: Examples.
% TikZ code for plotting in LaTeX is output to the console.

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



    e1 = create_me([0.3-2;0],2,[0.5;0.5]);%blue
    e2 = create_me([0-2;0],1,[0;3]);%red
    e3 = create_me([2-2;0],6);%green

    sc = 3;
    e1 = e1.raise(sc);    
    e2 = e2.raise(sc);    
    e3 = e3.raise(sc);    
    
    e12 = e1.pool(e2);
    
    figure;hold;
    axis('square');axis('equal');

    prior = create_me([0;0],1);
    plotGaussian(prior,[],'black, dashed','k--');
    
    plotGaussian(e1,[],'blue','b');
    
    plotGaussian(e2,[],'red','r');

    plotGaussian(e3,[],'green','g');

    plotGaussian(e12,[],'magenta','m');

    blue_green = exp(e1.llr(e3)),
    red_green = exp(e2.llr(e3)),
    magenta_green = exp(e12.llr(e3)),
    blue_red = exp(e1.llr(e2)),
    

end