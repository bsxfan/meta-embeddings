function model = createDPMM(W,B,crp)

    if nargin==0
        test_this();
        return;
    end

    cholW = chol(W);
    cholB = chol(B);
    dim = size(W,1);
    

    model.sample = @sample;
    
    function [X,Means,hlabels,counts] = sample(n)
        [labels,counts] = crp.sample(n);
        m = length(counts);
        hlabels = sparse(labels,1:n,true,m,n);
        Means = cholB\randn(dim,m);
        X = cholW\randn(dim,n) + Means*hlabels;
    end


end


function P = sampleP(dim,tame)
    R = rand(dim,dim-1)/tame;
    P = eye(dim) + R*R.';
end


function test_this()

    dim = 2;
    tame = 10;
    sep = 50;
    
    small = false;
    
    if small
        n = 8;
        ent = 3;
    else
        n = 1000;
        ent = 10;
    end
    
    
    crp = create_PYCRP([],0,ent,n);
    
    W = sep*sampleP(dim,tame);
    B = sampleP(dim,tame);
    
    
    model = createDPMM(W,B,crp);
    [X,Means,hlabels,counts] = model.sample(n);
    
    counts
    
    close all;
    plot(X(1,:),X(2,:),'.b',Means(1,:),Means(2,:),'*r');
    
    
    

end