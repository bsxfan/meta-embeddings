function calc = create_BXE_calculator(log_expectations,prior,poi)

    calc.BXE = @BXE;
    calc.get_tar_non = @get_tar_non;

    n = length(poi);
    spoi = sparse(poi);
    tar = bsxfun(@eq,spoi,spoi.');

    ntar = 0;
    nnon = 0;
    for k=1:n-1
        jj = k+1:n;
        tari = full(tar(k,jj));
        ntari = sum(tari);
        ntar = ntar + ntari;
        nnon = nnon + length(jj) - ntari;
    end
    
    if isempty(prior)
        prior = ntar/(ntar+nnon);
    end
    
    plo = log(prior) - log1p(-prior);

    
    function y = BXE(A,B)
        LEc = log_expectations(A,B);
        yt = 0;
        yn = 0;
        for i=1:n-1
            jj = i+1:n;
            AA = bsxfun(@plus,A(:,i),A(:,jj));
            BB = bsxfun(@plus,B(:,i),B(:,jj));
            tari = full(tar(i,jj));
            LE2 = log_expectations(AA,BB);
            llr = LE2 - LEc(i) - LEc(jj);
            log_post = plo + llr;
            yt = yt + sum(softplus(-log_post(tari)));
            yn = yn + sum(softplus(log_post(~tari)));
        end
        
        y = prior*yt/ntar + (1-prior)*yn/(nnon);
        
        
    end
    
    function [tars,nons] = get_tar_non(A,B)
        LEc = log_expectations(A,B);
        tars = zeros(1,ntar);
        nons = zeros(1,nnon);
        tcount = 0;
        ncount = 0;
        for i=1:n-1
            jj = i+1:n;
            AA = bsxfun(@plus,A(:,i),A(:,jj));
            BB = bsxfun(@plus,B(:,i),B(:,jj));
            tari = full(tar(i,jj));
            LE2 = log_expectations(AA,BB);
            llr = LE2 - LEc(i) - LEc(jj);
            
            llr_tar = llr(tari);
            count = length(llr_tar);
            tars(tcount+(1:count)) = llr_tar;
            tcount = tcount + count;

            llr_non = llr(~tari);
            count = length(llr_non);
            nons(ncount+(1:count)) = llr_non;
            ncount = ncount + count;
            
        end
        
        
    end
    
end

function y = softplus(x)
% y = log(1+exp(x));
    y = x;
    f = find(x<30);
    y(f) = log1p(exp(x(f)));
end