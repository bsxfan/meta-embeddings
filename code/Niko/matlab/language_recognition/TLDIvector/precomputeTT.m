function TT = precomputeTT(T,d,k,m)

    TT = zeros(k*k,m);
    ii = 1:d;
    for i=1:m
        Ti = T(ii,:);
        TT(:,i) = reshape(Ti.'*Ti,k*k,1);
        ii = ii + d;
    end

end