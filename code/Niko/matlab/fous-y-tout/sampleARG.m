function Z = sampleARG(a,B,n,X)

    if nargin==0
        test_this();
        return;
    end


    dim = length(a);
    
    if ~exist('X','var') || isempty(X)
        X = randn(dim,n);
    end
    
    
    diagB = diag(B);
    mu = a./diagB;
    Z = bsxfun(@plus,mu,bsxfun(@rdivide,X,sqrt(diagB)));
    
    B0 = bsxfun(@rdivide,B,diagB);
    
    
    
    for i=2:dim
        jj = 1:i-1;
        Z(i,:) = Z(i,:) - B0(i,jj)*Z(jj,:);
    end
    
end
    
function test_this
    

    dim = 2; 
    n = 5000;
    a = randn(dim,1);
    R = randn(dim,dim+1);
    B = R*R.';
    
    
    
    
    X = randn(dim,n);
    
    tic;Z1 = sampleARG(a,B,n,X);toc
    tic;Z2 = sampleChol(a,B,n,X);toc
    
    
    mu = B\a,
    mu1 = mean(Z1,2),
    mu2 = mean(Z2,2),
    C = inv(B),
    C1 = cov(Z1.',1),
    C2 = cov(Z2.',1),
    
    
    

    close all;
    
    mx = max(max(Z1(:)),max(Z2(:)));
    mn = min(min(Z1(:)),min(Z2(:)));
    
    
    subplot(1,2,1);plot(Z1(1,:),Z1(2,:),'.');title('AR');axis([mn,mx,mn,mx]);axis('square');
    subplot(1,2,2);plot(Z2(1,:),Z2(2,:),'.');title('Chol');axis([mn,mx,mn,mx]);axis('square');





end