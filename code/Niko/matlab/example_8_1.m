function example_8_1

    zdim = 2;
    xdim = 20;      %required: xdim > zdim
    nu = 100;       %required: nu >= 1, integer, DF
    fscal = 2;      %increase fscal to move speakers apart
    
    F = randn(xdim,zdim)*fscal;

    [HTPLDA,V] = create_HTPLDA(F,nu);

    z1 = randn(zdim,1);  %speaker 1 
    z2 = randn(zdim,1);  %speaker 2
    Z = [z1,z2,z2];      %1 of speaker 1 and 2 of speaker 2
    
    
    [R,lambda] = sample_speaker(Z,F,nu/2,[],true);


    [A,b] = HTPLDA.extractSGMEs(R);
    [A2,B] = HTPLDA.SGME2GME(A,b);
    
    
    
    GME1 = create_plain_GME(V*A2(:,1),V*reshape(B(:,1),zdim,zdim)*V.');
    GME2 = create_plain_GME(V*A2(:,2),V*reshape(B(:,2),zdim,zdim)*V.');
    GME3 = create_plain_GME(V*A2(:,3),V*reshape(B(:,3),zdim,zdim)*V.');
    
    
    
    close all;
    figure;
    hold;
    plotGaussian(zeros(zdim,1),eye(zdim),'black, dashed','k--');

    plotGaussian(GME1,[],'blue','b');
    plotGaussian(GME2,[],'red','r');
    plotGaussian(GME3,[],'red','r');
    plot(z1(1),z1(2),'b*');
    plot(z2(1),z2(2),'r*');
    axis('square');axis('equal');
    
    poi = [1 1 2];

    prior = create_PYCRP(0,[],2,3);
    %prior = create_PYCRP([],0,2,3);
    %prior = create_flat_partition_prior(length(poi));
    
    calc = create_partition_posterior_calculator(HTPLDA.log_expectations,prior,poi);
    f = calc.logPost(A,b);
    exp([f([1 1 2]), f([1 1 1]), f([1 2 3]), f([1 2 2]), f([1 2 1])])
    
    

end