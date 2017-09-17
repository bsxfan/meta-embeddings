function MVG_ME_example4
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



    e1 = create_me([-2;0],2,[0.5;-0.5]*3);%blue
    e2 = create_me([2;0],1,[0.5;0.5]*2);%red
    %e3 = create_me([2-1;0],6);%green

    sc = 1;
    e1 = e1.scale(sc);    
    e2 = e2.scale(sc);    
    %e3 = e3.scale(sc);    
    
    e12 = e1.pool(e2);
    %c12 = e1.convolve(e2);
    
    figure;hold;
    axis('square');axis('equal');

    %prior = create_me([0;0],1);
    %plotGaussian(prior,[],'black, dashed','k--');
    
    plotGaussian(e1,[],'blue','b');
    
    plotGaussian(e2,[],'red','r');


    plotGaussian(e12,[],'magenta','m');

    %plotGaussian(c12,[],'green','g');
    
    
    
    blue_red = exp(metaEmb_llr(e1,e2)),



end