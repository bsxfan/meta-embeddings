function MVG_ME_example2

    close all;

    function E = create_me(mu,sigma,R)
        dim = length(mu);
        I = eye(dim);
        if exist('R','var') && ~isempty(R)
            B = sigma*I+R*R.';
        else
            B = sigma*I;
        end
        E = create_plain_metaEmb(B*mu,B);
    end


    figure;hold;
    axis('square');axis('equal');
    
    
    prior = create_me([0;0],1);
    plotGaussian(prior,[],'black, dashed','k--');

    red1 = create_me([0;0],1,[2;1]);
    red2 = create_me([0.2;0],1,[2.5;-0.5]);
    plotGaussian(red1,[],'red','r');
    plotGaussian(red2,[],'red','r');

    
    blue1 = create_me([5+0;0],1,[2;1]);
    blue2 = create_me([5+0.2;0],1,[2.5;-0.5]);
    plotGaussian(blue1,[],'blue','b');
    plotGaussian(blue2,[],'blue','b');
    
    
    LR_red = exp(metaEmb_llr(red1,red2)),
    LR_blue = exp(metaEmb_llr(blue1,blue2)),
    

    

end