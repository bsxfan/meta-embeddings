function X = augment_i_vectors(T,K,Mu,C,X,Z)
% Augments i-vectors by adding noise drawn from i-vector posterior covariance.
% The noise is projected into the complement of the subspace spanned by the
% language means, so that it destoyes no language information. The noise in
% the complement space `signals' the magnitude of the posterior covariance 
% to the t-distribution backend.
%
% Inputs:
%   TT: dim^2-by-K, where dim is i-vector dimension and K UBM size. 
%       TT = precomputeTT(T,...)
%   Mu: dim-by-L language means, for L languages
%   C: within language covariance
%   X: dim-by-N, matrix of N i-vectors
%   Z: K-by-N zero-order stats
%
% Outputs:
%   X: augmented i-vectors
%
% Typical usage:
%  Given: i-vectors: Train, Test
%         zero-order stats: Ztrain, Ztest
%         extractor T-matrix (dim*fdim)-by-K 
%  > TT = precomputeTT(T,fdim,dim,K);
%  > Train0 = Train;
%  > TBE = create_T-backend(nu,dim,K);  %nu is fixed by user and not learnt during training
%  Iterate a few times:
%    > TBE.train(Train,Labels,10); % TrainData: dim-by-N, Labels: K-by-N, (sparse) one-hot labels
%    > [Mu,C] = TBE.getParams();
%    > augment = augment_i_vectors(TT,Mu,C);
%    > Train = augment(Train0,Ztrain);
%  Test = augment(test,Ztest);
%  Train_LLH = TBE.logLH(Train);  
%  Test_LLH = TBE.logLH(Test);


    R = chol(C);   %R'*R = C
    
    [dim,L] = size(Mu);
    dim2 = dim^2;
    
    TT = zeros(dim2,m);
    ii = 1:d;
    for k=1:m
        Tk = R'\T(ii,:);
        TT(:,k) = (dim/(dim-L))*reshape(Tk.'*Tk,dim2,1);
        ii = ii + d;
    end

    
    Mu = R'\Mu;

    Q = orth(Mu);
    Nap = eye(size(Q)) - Q*Q.';
    
    if ~exist('X','var') || isempty(X)
        X = @augment;
    else
        X = augment(X,Z);
    end
    
    
    function X = augment(X,Z)
    
        B = TT*Z;  %vectorized posterior precisions

        X = R'\X;

        N = size(X,2);
        
        Noise = randn(dim,N);
        for i=1:N
            Bi = reshape(B(:,i),dim,dim);
            cholB = chol(Bi);
            Noise(:,i) = cholB'*Noise(:,i); 
        end
        X = R'*(X+Nap*Noise);
    end
    

end