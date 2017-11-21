function example_8_1

    zdim = 2;
    xdim = 20;      %required: xdim > zdim
    nu = 100;         %required: nu >= 1, integer, DF
    fscal = 1/2;
    
    F = randn(xdim,zdim)/fscal;

    z1 = randn(zdim,1);  %speaker 1 
    z2 = randn(zdim,1);  %speaker 2
    Z = [z1,z2,z2];      %1 of speaker 1 and 2 of speaker 2
    
    
    [X,lambda] = sample_speaker(Z,F,nu/2,[],true);


    B = F'*F;
    BF = B\F.';
    G = eye(xdim) - F*BF;
    nu_prime = nu + ( xdim - zdim );
    
    E = sum(X.*(G*X),1);
    beta = nu_prime./(nu+E);
    Mu = BF*X;
    
    close all;
    figure;
    hold;
    plotGaussian(zeros(zdim,1),eye(zdim),'black, dashed','k--');

    plotGaussian(Mu(:,1),inv(beta(1)*B),'blue','b');
    plotGaussian(Mu(:,2),inv(beta(2)*B),'red','r');
    plotGaussian(Mu(:,3),inv(beta(3)*B),'red','r');
%     plotGaussian(Mu(:,1),inv(lambda(1)*B),'blue',':b');
%     plotGaussian(Mu(:,2),inv(lambda(2)*B),'red',':r');
%     plotGaussian(Mu(:,3),inv(lambda(3)*B),'red',':r');
    plot(z1(1),z1(2),'b*');
    plot(z2(1),z2(2),'r*');
    axis('square');axis('equal');

end