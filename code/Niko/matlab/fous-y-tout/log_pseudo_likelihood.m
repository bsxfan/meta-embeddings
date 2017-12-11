function [y,back] = log_pseudo_likelihood(A,B,d,logPrior,num)
        
    %original table stats
    At = A*blocks.';
    Bt = B*blocks.';
    %log-expectations for every original table
    [LEt,back1] = log_expectations(At,Bt);

    %log-expectations for every customer, alone at singleton table
    [LEc,back2] = log_expectations(A,B);  


    %For every customer, the stats for the table after that customer has
    %just left.
    Amin = bsxfun(@minus,At(:,poi),A);
    Bmin = bsxfun(@minus,Bt(:,poi),B);
    [LEmin,back3] = log_expectations(Amin,Bmin);


    for i=1:m
        tar = full(blocks(i,:));
        non = ~tar;

        %non-targets
        Aplus = bsxfun(@plus,A(:,non),At(:,i));
        Bplus = bsxfun(@plus,B(:,non),Bt(:,i));
        LLR(i,non) = log_expectations(Aplus,Bplus) - LEt(i) - LEc(non);

        %targets
        LLR(i,tar) = LEt(i) - LEmin(tar) - LEc(tar);

    end

    logPost = LLR + logPrior;
    [y,back4] = sumlogsoftmax(logPost,num);


    back = @back_this;


    function [dA,dB,dd] = back_this(dy)


    end



end
