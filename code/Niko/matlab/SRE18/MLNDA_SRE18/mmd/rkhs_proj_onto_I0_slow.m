function [y,back] = rkhs_proj_onto_I0_slow(sigma,A,b,B)
% RKHS inner products of multivariate Gaussians onto standard normal. The
% RKHS kernel is K(x,y) = ND(x|y,sigma I). The first order natural
% parameters of the Gaussians are in the columns of A. The precisions are
% proportional to the fixed B, with scaling contants in the vector b.


    if nargin==0
        test_this();
        return;
    end

    [dim,n] = size(A);

    Bconv = eye(dim)/(1+sigma);
    aconv = zeros(dim,1 );
    
    lgn1 = log_gauss_norm(Bconv,aconv);  % (-dim/2)*log1p(sigma);
    
    lgn2 = zeros(1,n);
    lgn12 = zeros(1,n);
    for i=1:n
        lgn2(i) = log_gauss_norm(b(i)*B,A(:,i));
        lgn12(i) = log_gauss_norm(Bconv + b(i)*B,A(:,i));
    end
    
    y = exp(lgn1 + lgn2 - lgn12);
    
    back = @back_this;
    
    
    function [dA,db,dB] = back_this(dy)
        
        dlgn2 = dy.*y;
        dlgn12 = -dlgn2;
        
        dA = zeros(dim,n);
        db = zeros(1,n);
        dB = zeros(size(B));
        for ii=1:n
            bB = b(ii)*B;
            a = A(:,ii);
            [~,back1] = log_gauss_norm(bB,a);
            [dbB,da] = back1(dlgn2(i));
            dA(:,ii) = da;
            db(ii) = dbB(:).'*B(:);
            dB = dB + b(ii)*dbB;
            
            [~,back2] = log_gauss_norm(Bconv + bB,a);
            [dbB,da] = back2(dlgn12(ii));
            dA(:,ii) = dA(:,ii) + da;
            db(ii) = db(ii) + dbB(:).'*B(:);
            dB = dB + b(ii)*dbB;

        end
        
    end
    


end


function test_this()

    dim = 4;
    n = 3;
    A = randn(dim,n);
    b = randn(1,n).^2;
    B = randn(dim);B=B*B.';
    
    sigma = pi;

    f = @(A,b,B) rkhs_proj_onto_I0_slow(sigma,A,b,B);

    testBackprop(f,{A,b,B},{0,0,1});
    
end