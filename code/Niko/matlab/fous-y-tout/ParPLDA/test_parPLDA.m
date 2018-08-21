function test_parPLDA()

    xdim = 10;
    ydim = 20;
    zdim = 5;
    cdim = 5;
    
    
    szx = 3;
    szy = 6;
    scx = 2;
    scy = 2;
    
    Vx = szx*randn(xdim,zdim);
    Vy = szy*randn(ydim,zdim);
    
    Rx = randn(xdim,xdim);
    Ry = randn(ydim,ydim);

    Wx = scx*randn(xdim,cdim);
    Wy = scy*randn(ydim,cdim);
    mux = randn(xdim,1);
    muy = randn(ydim,1);
    
    
    
    model0.V = [Vx;Vy];
    model0.W = inv([Rx*Rx.'+Wx*Wx.', Wx*Wy.'; Wy*Wx.', Ry*Ry.'+Wy*Wy.']);
    model0.mu = [mux;muy];
    model0 = derive_parPLDA(model0,xdim,ydim);
    
    ntrain = 10000;
    ntrainsp = 100;
    [X,Y,Labels] = sample(ntrain,ntrainsp);
    model = train_parPLDA(X,Y,Labels,zdim);
    
    
    ntest = 100000;
    [Xe,Ye,Z] = sample_single(ntest);
    [Xt,Yt] = sample_single(Z);
    [Xn,Yn] = sample_single(ntest);
    
    Ex = model.extractME(Xe,[]);
    Ey = model.extractME([],Ye);
    Exy = model.extractME(Xe,Ye);
    
    Tx = model.extractME(Xt,[]);
    Ty = model.extractME([],Yt);
    Txy = model.extractME(Xt,Yt);
    
    Nx = model.extractME(Xn,[]);
    Ny = model.extractME([],Yn);
    Nxy = model.extractME(Xn,Yn);
    

    Ex0 = model0.extractME(Xe,[]);
    Ey0 = model0.extractME([],Ye);
    Exy0 = model0.extractME(Xe,Ye);
    
    Tx0 = model0.extractME(Xt,[]);
    Ty0 = model0.extractME([],Yt);
    Txy0 = model0.extractME(Xt,Yt);
    
    Nx0 = model0.extractME(Xn,[]);
    Ny0 = model0.extractME([],Yn);
    Nxy0 = model0.extractME(Xn,Yn);
    
    
    tar = model.scoreTrials(Ex,Tx);
    non = model.scoreTrials(Ex,Nx);
    tar0 = model0.scoreTrials(Ex0,Tx0);
    non0 = model0.scoreTrials(Ex0,Nx0);
    EERxx = 100*[eer(tar,non),eer(tar0,non0)],
    
    tar = model.scoreTrials(Ex,Ty);
    non = model.scoreTrials(Ex,Ny);
    tar0 = model0.scoreTrials(Ex0,Ty0);
    non0 = model0.scoreTrials(Ex0,Ny0);
    EERxy = 100*[eer(tar,non),eer(tar0,non0)],
    
    tar = model.scoreTrials(Ey,Ty);
    non = model.scoreTrials(Ey,Ny);
    tar0 = model0.scoreTrials(Ey0,Ty0);
    non0 = model0.scoreTrials(Ey0,Ny0);
    EERyy = 100*[eer(tar,non),eer(tar0,non0)],
    
    tar = model.scoreTrials(Exy,Tx);
    non = model.scoreTrials(Exy,Nx);
    tar0 = model0.scoreTrials(Exy0,Tx0);
    non0 = model0.scoreTrials(Exy0,Nx0);
    EER2x = 100*[eer(tar,non),eer(tar0,non0)],
    
    tar = model.scoreTrials(Exy,Ty);
    non = model.scoreTrials(Exy,Ny);
    tar0 = model0.scoreTrials(Exy0,Ty0);
    non0 = model0.scoreTrials(Exy0,Ny0);
    EER2y = 100*[eer(tar,non),eer(tar0,non0)],
    
    tar = model.scoreTrials(Exy,Txy);
    non = model.scoreTrials(Exy,Nxy);
    tar0 = model.scoreTrials(Exy0,Txy0);
    non0 = model.scoreTrials(Exy0,Nxy0);
    EER22 = 100*[eer(tar,non),eer(tar0,non0)],

    
    
    
    function [X,Y,Labels,Z] = sample(n,nsp)
        sp = randi(nsp,1,n);
        Labels = sparse(sp,1:n,true,nsp,n);    %nsp-by-n
        Labels(sum(Labels,2)==0,:) = [];
        nsp = size(Labels,1);
        Z = randn(zdim,nsp);
        ZZ = Z*Labels;
        
        C = randn(cdim,n);
        X = bsxfun(@plus,mux,Rx*randn(xdim,n) + Wx*C + Vx*ZZ);
        Y = bsxfun(@plus,muy,Ry*randn(ydim,n) + Wy*C + Vy*ZZ);
    end

    function [X,Y,Z] = sample_single(Z)
        if isscalar(Z)
            n = Z;
            Z = randn(zdim,n);
        else
            n = size(Z,2);
        end
        
        C = randn(cdim,n);
        X = Rx*randn(xdim,n) + Wx*C + Vx*Z;
        Y = Ry*randn(ydim,n) + Wy*C + Vy*Z;
    end










end