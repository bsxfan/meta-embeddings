function test_parPLDA()
% Uses synthetic data to test and demo `paralell PLDA', which can switch 
% between say i-vectors and x-vectors from enroll to test.
%
% See train_parPLDA for full documentation.

    xdim = 10;      % i-vector dimension
    ydim = 20;      % x-vector dimension 
    zdim = 5;       % speaker factor dimension
    
    cdim = 5;       % dimension of another hidden factor, common to both types of vectors,
                    % which makes them dependent, even when the speaker
                    % factor is given.
    
    
    % For the synthetic data generation, some model parameter scaling 
    % factors to get realistic EERs:
    szx = 3;   % i-vector signal to noise ratio: increase this to improve i-vector EER
    szy = 6;   % x-vector signal to noise ratio: increase this to improve x-vector EER
    scx = 2;   % strength of dependency between common factor and i-vectors
    scy = 2;   % strength of dependency between common factor and i-vectors
    
    % speaker factor loading matrices
    Vx = szx*randn(xdim,zdim);
    Vy = szy*randn(ydim,zdim);
    
    % These induce channel noise auto-correlations
    Rx = randn(xdim,xdim);
    Ry = randn(ydim,ydim);

    % These induce common factor to data correlations
    Wx = scx*randn(xdim,cdim);
    Wy = scy*randn(ydim,cdim);
    
    % data means
    mux = randn(xdim,1);
    muy = randn(ydim,1);
    
    
    % For oracle scoring: the model that created the data
    model0.V = [Vx;Vy];
    model0.W = inv([Rx*Rx.'+Wx*Wx.', Wx*Wy.'; Wy*Wx.', Ry*Ry.'+Wy*Wy.']);
    model0.mu = [mux;muy];
    model0 = derive_parPLDA(model0,xdim,ydim);

    % For a scoring sanity check: the oracle model limited to only the X
    % part.
    modelX.V = Vx;
    modelX.W = inv(Rx*Rx.'+Wx*Wx.');
    modelX.mu = mux;
    modelX = SPLDA_equip_with_extractor(modelX);
    modelX = equip_with_GME_scoring(modelX,zdim);
    
    
    % Sample synthetic training data and train model
    ntrain = 10000;
    ntrainsp = 100;
    [X,Y,Labels] = sample(ntrain,ntrainsp);
    model = train_parPLDA(X,Y,Labels,zdim);
    
    
    fprintf('Test trained vs ideal models, with various combinations of X, Y and [X;Y] in enroll and test.\n');
    fprintf('See code documentation for explanations\n');
    fprintf('Testing ''scoreTrials()'':\n');
    
    ntest = 100000;
    [Xe,Ye,Z] = sample_single(ntest);    % sample enrollment i-vectors and x-vectors, all from different speakers 
    [Xt,Yt] = sample_single(Z);          % sample target test trials, from the same speakers
    [Xn,Yn] = sample_single(ntest);      % sample non-target test trials, from new speakers
    
    % extract enrollment meta-embeddings, using different combinations
    Ex = model.extractME(Xe,[]);
    Ex_alt = modelX.extractME(Xe);
    Ey = model.extractME([],Ye);
    Exy = model.extractME(Xe,Ye);
    Exy_alt = model.poolME(Ex,Ey);   %suboptimal: **not** the same as Exy, because X,Y|Z are not independent
    
    % extract target test meta-embeddings, using different combinations
    Tx = model.extractME(Xt,[]);
    Tx_alt = modelX.extractME(Xt);  %for sanity check
    Ty = model.extractME([],Yt);
    Txy = model.extractME(Xt,Yt);
    
    % extract non-target test meta-embeddings, using different combinations
    Nx = model.extractME(Xn,[]);
    Nx_alt = modelX.extractME(Xn);
    Ny = model.extractME([],Yn);
    Nxy = model.extractME(Xn,Yn);
    

    %extract everything using oracle model too
    Ex0 = model0.extractME(Xe,[]);
    Ey0 = model0.extractME([],Ye);
    Exy0 = model0.extractME(Xe,Ye);
    
    Tx0 = model0.extractME(Xt,[]);
    Ty0 = model0.extractME([],Yt);
    Txy0 = model0.extractME(Xt,Yt);
    
    Nx0 = model0.extractME(Xn,[]);
    Ny0 = model0.extractME([],Yn);
    Nxy0 = model0.extractME(Xn,Yn);
    
    % compute EER for enroll and test using only i-vectors
    tar = model.scoreTrials(Ex,Tx);
    non = model.scoreTrials(Ex,Nx);
    tar0 = model0.scoreTrials(Ex0,Tx0);
    non0 = model0.scoreTrials(Ex0,Nx0);
    tar1 = modelX.scoreTrials(Ex_alt,Tx_alt);
    non1 = modelX.scoreTrials(Ex_alt,Nx_alt);
    % [trained model, oracle model, sanity (should = oracle)]
    EERxx = 100*[eer(tar,non),eer(tar0,non0),eer(tar1,non1)],
    CLLRxx = cllr(tar,non),
    
    % compute EER for enroll with i-vectors and test with x-vectors
    tar = model.scoreTrials(Ex,Ty);
    non = model.scoreTrials(Ex,Ny);
    tar0 = model0.scoreTrials(Ex0,Ty0);
    non0 = model0.scoreTrials(Ex0,Ny0);
    % [trained model, oracle model]
    EERxy = 100*[eer(tar,non),eer(tar0,non0)],
    
    % compute EER for enroll and test using only x-vectors
    tar = model.scoreTrials(Ey,Ty);
    non = model.scoreTrials(Ey,Ny);
    tar0 = model0.scoreTrials(Ey0,Ty0);
    non0 = model0.scoreTrials(Ey0,Ny0);
    % [trained model, oracle model]
    EERyy = 100*[eer(tar,non),eer(tar0,non0)],
    
    % compute EER for enroll with both and test with x-vectors
    tar = model.scoreTrials(Exy,Tx);
    non = model.scoreTrials(Exy,Nx);
    tar1 = model.scoreTrials(Exy_alt,Tx);  % suboptimal, incorrect enrollment pooling
    non1 = model.scoreTrials(Exy_alt,Nx);  % suboptimal, incorrect enrollment pooling
    tar0 = model0.scoreTrials(Exy0,Tx0);
    non0 = model0.scoreTrials(Exy0,Nx0);
    % [trained model, incorrect enrollment pooling, oracle model]
    EER2x = 100*[eer(tar,non),eer(tar1,non1),eer(tar0,non0)],
    
    % compute EER for enroll with both and test with y-vectors
    tar = model.scoreTrials(Exy,Ty);
    non = model.scoreTrials(Exy,Ny);
    tar0 = model0.scoreTrials(Exy0,Ty0);
    non0 = model0.scoreTrials(Exy0,Ny0);
    % [trained model, oracle model]
    EER2y = 100*[eer(tar,non),eer(tar0,non0)],
    
    % compute EER for enroll with both and test with both
    tar = model.scoreTrials(Exy,Txy);
    non = model.scoreTrials(Exy,Nxy);
    tar0 = model.scoreTrials(Exy0,Txy0);
    non0 = model.scoreTrials(Exy0,Nxy0);
    % [trained model, oracle model]
    EER22 = 100*[eer(tar,non),eer(tar0,non0)],
    CLLR22 = cllr(tar,non),

    
    fprintf('\n\n\nTesting ''scoreMatrix()'':\n');
    nenroll = 800;
    ntest = 1200;
    enr = 1:nenroll;
    tst = nenroll+(1:ntest);
    nsp = 200;
    [X,Y,Labels] = sample(nenroll+ntest,nsp);
    Xe = X(:,enr);
    Ye = Y(:,enr);
    Le = Labels(:,enr);
    Xt = X(:,tst);
    Yt = Y(:,tst);
    Lt = Labels(:,tst);
    
    Tar = logical(Le.'*Lt);   % nenroll-by-ntest

    Ex = model.extractME(Xe,[]);
    Ey = model.extractME([],Ye);
    Exy = model.extractME(Xe,Ye);

    Tx = model.extractME(Xt,[]);
    Ty = model.extractME([],Yt);
    Txy = model.extractME(Xt,Yt);
    
    LLR = model.scoreMatrix(Ex,Tx);
    tar = LLR(Tar);
    non = LLR(~Tar);
    EERxx = 100*eer(tar,non),
    CLLRxx = cllr(tar,non),

    LLR = model.scoreMatrix(Ex,Ty);
    tar = LLR(Tar);
    non = LLR(~Tar);
    EERxy = 100*eer(tar,non),

    LLR = model.scoreMatrix(Ey,Ty);
    tar = LLR(Tar);
    non = LLR(~Tar);
    EERyy = 100*eer(tar,non),
    
    LLR = model.scoreMatrix(Exy,Tx);
    tar = LLR(Tar);
    non = LLR(~Tar);
    EER2x = 100*eer(tar,non),
    

    LLR = model.scoreMatrix(Exy,Ty);
    tar = LLR(Tar);
    non = LLR(~Tar);
    EER2y = 100*eer(tar,non),
    
    LLR = model.scoreMatrix(Exy,Txy);
    tar = LLR(Tar);
    non = LLR(~Tar);
    EER22 = 100*eer(tar,non),
    CLLR22 = cllr(tar,non),
    
    
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