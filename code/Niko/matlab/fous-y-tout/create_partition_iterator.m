function PI = create_partition_iterator(n)
% Iterate through all the partitions(n-th Bell number of them) of a set of n elements.
%
% The algorithm is from:
% @techreport{orlov,
%   author = {Michael Orlov}, title = {Efficient Generation of Set Partitions},
% 	institution = {Computer Science Department of {B}en-{G}urion {U}niversity in {I}srael},
% 	year = {2002}, month = {March},
% 	url = {http://www.cs.bgu.ac.il/~orlovm/papers/partitions.pdf}
% }
%
% This code implements algorithms 1 and 3 of that paper. The 
% algorithms have been adapted for 1-based indexing.
%
% Input: n>=1, the set size.
% Output: PI, the iterator, a struct with one function handle:
%   labels = PI.next() returns the next partition, where
%     labels: n-vector of block indices in the range 1...n. All set members
%             that share the same block index are in the same block (subset).
%     labels = [] signals that all possibilities have been exhausted 
%
% Note: The coarsest partition, labels = ones(1,n) is returned first and the finest,
%       labels = [1,2,...,n] last. (See the paper for iteration in the
%       reverse direction.)
%


    labels = ones(1,n);
    m = ones(1,n);
    
    PI.next = @next;
    
    function labels1 = next()
        labels1 = labels;
        if isempty(labels)
            return;
        end
        for i=n:-1:2
            if labels(i) <= m(i-1)
                labels(i) = labels(i) + 1;
                m(i) = max(m(i),labels(i));
                labels(i+1:end) = 1;
                m(i+1:end) = m(i);
                return;
            end
        end
        labels = [];
    end
    



end