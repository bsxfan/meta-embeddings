function test_parPLDA()

    xdim = 10;
    ydim = 20;
    zdim = 5;
    cdim = 5;
    
    
    szx = 5;
    szy = 10;
    scx = 1;
    scy = 1;
    
    Vx = szx*randn(xdim,zdim);
    Vy = szy*randn(ydim,zdim);
    
    Rx = randn(xdim,xdim);
    Ry = randn(ydim,ydim);

    Wx = scx*randn(xdim,cdim);
    Wy = scy*randn(ydim,cdim);
    
    
    ntrain = 10000;
    ntrainsp = 100;
    [X,Y,Labels] = sample(ntrain,ntrainsp);
    model = train_parPLDA(X,Y,Labels,zdim);
    
    
    
    function [X,Y,Labels] = sample(n,nsp)
        sp = randi(nsp,1,n);
        Labels = sparse(sp,1:n,true,nsp,n);    %nsp-by-n
        Labels(sum(Labels,2)==0,:) = [];
        nsp = size(Labels,1);
        Z = randn(zdim,nsp);
        ZZ = Z*Labels;
        
        C = randn(cdim,n);
        X = Rx*randn(xdim,n) + Wx*C + Vx*ZZ;
        Y = Ry*randn(ydim,n) + Wy*C + Vy*ZZ;
    end











end