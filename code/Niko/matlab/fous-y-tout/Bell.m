function [b B] = Bell(n)
% b = Bell(N)
% N integer
% Compute the Bell's number. It is the number of all possible partitions
% of the set {1:N}.
% [b B] = Bell(N) returns in column vector B all Bell's numbers from 1 to N
%
% Author: Bruno Luong <brunoluong@yahoo.com>
% History
%   Original: 17-May-2009
%   Last update: 18-May-2009, cosmetic changes

n=double(n);
if n==0
    b = 1;
    B = zeros(0,1);
    return
end
A = zeros(n);
A(1,1) = 1;
for i=2:n
    A(i,1) = A(i-1,i-1);
    for j=2:n
        A(i,j) = A(i,j-1) + A(i-1,j-1);
    end
end
b = A(n,n);
if nargout>=2
    B = diag(A);
end
end