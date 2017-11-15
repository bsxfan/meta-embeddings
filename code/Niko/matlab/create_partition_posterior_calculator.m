function C = create_partition_posterior_calculator(n,prior)
    
    m = 2^n-1;
    subsets = logical(mod(fix(bsxfun(@rdivide,0:m,2.^(0:n-1)')),2));
    subsets = subsets(:,2:end);  % dump empty subset

    C.logPost = @logPost;
    
    
    % A,B: n-column matrices of natural parameters for n meta-embeddings
    function y = logPost(A,B)
    
        [dim,n1] = size(A);
        assert(n1==n);
        assert(size(B,2)==n);


        % accumulate natural params for every subset
        A = A*subsets;  
        B = B*subsets;
        log_ex = zeros(1,m); 
        for i=1:m
            E = create_plain_GME(A(:,i),reshape(B(:,i),dim,dim),0);
            log_ex(i) = E.log_expectation();
        end
    
    end


end