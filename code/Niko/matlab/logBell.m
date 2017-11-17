function b = logBell(n)
% b = logBell(n)
% n integer
% Compute the log Bell number. It is the number of all possible partitions
% of the set {1:N}.
%
% Complexity is quadratic in n. See also approx_log_Bell 
%
if n==0 || n==1
    b = 0;
    return
end
A1 = zeros(1,n);
A2 = zeros(1,n);
A1(1) = 0;
for i=2:n
    A2(1) = A1(i-1);
    for j=2:n
        a2 = A2(j-1);
        a1 = A1(j-1);
        if a2 > a1
            A2(j) = a2 + log1p(exp(a1-a2));
        else
            A2(j) = a1 + log1p(exp(a2-a1));
        end
        %A2(j) = A2(j-1) + A1(j-1);
    end
    A1 = A2;    
end
b = A2(n);
end