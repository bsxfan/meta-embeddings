function D = randDirichlet(alpha,m,n)
    D = randg(alpha,m,n);
    D = bsxfun(@rdivide,D,sum(D,1));
end