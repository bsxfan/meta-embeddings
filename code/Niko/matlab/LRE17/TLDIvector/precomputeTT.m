function TT = precomputeTT(T,d,k,m)

    TT = zeros(k*k,m);
    ii = 1:d;
    for i=1:m
        Ti = T(ii,:);
        TT(:,i) = Ti.'*Ti;
        ii = ii + d;
    end

end