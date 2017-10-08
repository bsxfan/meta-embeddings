function R = randDirichlet(alpha,m,n)
% Generates m-by-n matrix of n samples from m-category Dirichlet, with
% concentration parameter: alpha > 0.

    if nargin==0
        test_this();
        return;
    end

    R = reshape(randgamma(alpha,1,m*n),m,n);
    R = bsxfun(@rdivide,R,sum(R,1));


end

function test_this()

    close all;
    m = 100;
    alpha = 1/(2*m);
    
    n = 5000;
    R = randDirichlet(alpha,m,n);
    maxR = max(R,[],1);
    hist(maxR,100);

%      n = 50;
%      R = randDirichlet(alpha,m,n);
%      hist(R(:),100);
end

