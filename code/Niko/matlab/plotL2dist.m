function [D,a,b] = plotL2dist

    x = create_plain_metaEmb(0,10);  %prior for now
    
    a = -10:0.1:10;
    b = 0.1:10;
    
    D = zeros(length(b),length(a));
    
    for i=1:length(b);
        for j=1:length(a);
            y = create_plain_metaEmb(a(j),b(i));
            y = y.L1normalize();
            D(i,j) = sqrt(x.distance_square(y));
        end
    end
    
    



end