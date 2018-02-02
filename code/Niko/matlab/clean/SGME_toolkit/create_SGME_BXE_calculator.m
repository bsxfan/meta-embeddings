function [bxe,tar_non] = create_SGME_BXE_calculator(labels)


    spoi = sparse(labels);
    tar = bsxfun(@eq,spoi,spoi.');
    n = length(labels);
    
    ntar = (sum(tar(:)) - n )/2;
    ntot = n*(n-1)/2;
    nnon = ntot - ntar; 

%     ntar = 0;
%     nnon = 0;
%     for k=1:n-1
%         jj = k+1:n;
%         tari = full(tar(k,jj));
%         ntari = sum(tari);
%         ntar = ntar + ntari;
%         nnon = nnon + length(jj) - ntari;
%     end
    
    prior = ntar/ntot;
    plo = log(prior) - log1p(-prior);
    
    wt = prior/(ntar*log(2));
    wn = (1-prior)/(nnon*log(2));

    bxe = @(E,D) addME_BXE(E,D,@SGME_logexpectations,plo,wt,wn,tar);

    tar_non = @get_tar_non; 
    
    
    function [tars,nons] = get_tar_non(E,D)
        LEc = SGME_logexpectations(E,D);
        tars = zeros(1,ntar);
        nons = zeros(1,nnon);
        tcount = 0;
        ncount = 0;
        for i=1:n-1
            jj = i+1:n;
            EE = bsxfun(@plus,E(:,i),E(:,jj));
            tari = full(tar(i,jj));
            LE2 = SGME_logexpectations(EE,D);
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