function create_fast_CollapsedGibbs(X,F,W,alpha,hlabels,ssz)

    [V,Lambda] = eig(F'*W*F);
    Lambda = diag(Lambda);
    F = F*V;
    P = F.'*W;
    A = P*X;
    A2 = A.^2;
    
    n = size(A,2);
    logDets = sum(log1p(Lambda*(1:n)),1);
    gammaLogs = gammaln(alpha+(1:n));
    delta = gammaLogs - logDets/2;


    %logP = logPtarget(hlabels);
    single_scores = ( sum(bsxfun(@rdivide,A2,1+Lambda),1) - logDets(1) )/2;  %1-by-n

    counts = sum(hlabels,2);
    Q = 1 + Lambda*counts.';   %d-by-m
    AL = A*hlabels.';
    full_scores = ( sum(AL.^2) - logDets(counts) ) /2;  %1-by-m
    logP = sum(gammaLogs(counts)) - sum(fulll_scores);
        
    [ilabels,~] = find(labels);
    Arem = AL(:,ilabels) - A;
    rem_counts = counts(ilabels) -1;
    Qrem = 1 + Lambda*rem_counts.';   %d-by-m
    rem_scores = ( sum(Arem.^2./Qrem,1) - logDets(rem_counts) )/2;    %1-by-n
    
    
    Qplus = 1 + Lambda*(1+counts.'); 
    Left = AL./Qplus;
    Cplus = 0.5./Qplus;
    sumLL = sum(Left.*AL,1)/2;
    
    %rest = true(1:n);
    
        
    function logP = logPtarget(hlabels)
        counts = sum(hlabels,2);
        Q = 1 + Lambda*counts.';   %d-by-m
        AL = A*hlabels.';
        logP = sum(delta(counts)) + sum(AL(:).^2./Q(:))/2;
    end
        

    function [hlabels,newlogP] = iteration()
        
        %select set of `test' labels to re-sample
        vv = unique(randi(n,1,ssz)); % might be less than ssz (but at least one)
        %rest(vv) = false;
        
        %score all original `enrollment' (nothing removed) against all `tests'
        Scores = bsxfun(@plus,Left.'*A(:,vv)+Cplus.'*A2,sumLL.'-fullscores.');  % m-by-ssz
        Scores = bsxfun(@minus,Scores,single_scores(vv));
        % we still need to subtract logDet terms!
        
        
        %We need to correct `self-scores', where test labels are already
        %present in the enrollment set. There are only length(vv) such
        %scores.
        scores = full_scores(ilabels(vv)) - single_scores(vv) - rem_scores(vv);
        
        
        
        %rest(vv) = true;
    end
        
        


end

