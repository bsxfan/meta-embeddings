function [subsets,counts] = labels2blocks(labels)
% Inputs:
%   labels: n-vector with elements in 1..m, maps each of n customers to a
%           table number. There are m tables. Empty tables not allowed. 
%
% Ouputs:
%   subsets: n-by-m logical, with one-hot rows
%   counts: m-vector, maps table number to customer count

    if nargin==0
        test_this();
        return;
    end

    m = max(labels);    %m tables
    n = length(labels); %n customers
    assert(min(labels)==1,'illegal argument ''labels'': tables must be consecutively numbered from 1');
    assert(m <= n,'illegal argument ''labels'': there are more tables than customers');
    
    subsets = bsxfun(@eq,1:m,labels(:));
    %subsets = sparse(1:n,labels,true,n,m,n);
    counts = sum(subsets,1);

    assert(sum(counts)==n,'illegal argument ''labels'': table counts must add up to length(labels)');
    assert(all(counts),'illegal argument ''labels'': empty tables not allowed');

end

function test_this()

    labels = [2,3,3,3,4];
    [subsets,counts] = labels2blocks(labels)
end