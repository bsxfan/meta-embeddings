function y = plain_GME_log_expectations(A,B)

    [dim,ns] = size(A);
    y = zeros(1,ns);
    for i=1:ns
        E = create_plain_GME(A(:,i),reshape(B(:,i),dim,dim),0);
        y(i) = E.log_expectation();
    end
        
        
end
