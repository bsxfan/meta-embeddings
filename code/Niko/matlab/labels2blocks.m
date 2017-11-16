function [subsets,counts] = labels2blocks(labels)

    if nargin==0
        test_this();
        return;
    end

    m = max(labels);    %m tables
    n = length(labels); %n customers
    assert(min(labels)==1,'illegal argument ''labels'': tables must be consecutively numbered from 1');
    assert(m <= n,'illegal argument ''labels'': there are more tables than customers');
    
%     subsets = false(n,m);
%     for j=1:m
%         ii = labels==j;
%         assert(any(ii),'illegal argument ''labels'': tables must be consecutively numbered from 1');
%         subsets(ii,j) = true;
%     end
    subsets = bsxfun(@eq,1:m,labels(:));
    counts = sum(subsets,1);
    assert(sum(counts)==n,'illegal argument ''labels'': table counts must add up to length(labels)');
    assert(all(counts),'illegal argument ''labels'': empty tables not allowed');

end

function test_this()

    labels = [2,3,3,3,4];
    [subsets,counts] = labels2blocks(labels)
end