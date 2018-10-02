function test_MLNDA()

    % Assemble model to generate data
    big = false;
    nu = inf;           %required: nu >= 1, integer, degrees of freedom for heavy-tailed channel noise
    if ~big
        zdim = 2;       %speaker identity variable size 
        rdim = 20;      %i-vector size. required: rdim > zdim
        fscal = 3;      %increase fscal to move speakers apart
    else
        zdim = 100;       %speaker identity variable size 
        rdim = 512;      %i-vector size. required: rdim > zdim
        fscal = 1/20;      %increase fscal to move speakers apart
    end
    
    
    
    F = randn(rdim,zdim)*fscal;
    W = randn(rdim,2*rdim); W = W*W.';W = (rdim/trace(W))*W;
    %model1 = create_HTPLDA_SGME_backend(nu,F,W);  %oracle model
    
    
    %Generate synthetic labels
    nspeakers = 100;
    recordings_per_speaker = 10;
    N = nspeakers*recordings_per_speaker;
    ilabels = repmat(1:nspeakers,recordings_per_speaker,1);
    ilabels = ilabels(:).';  % integer speaker labels
    hlabels = sparse(ilabels,1:N,true,nspeakers,N); %speaker label matrix with one-hot columns
    
    %and some training data
    Z = randn(zdim,nspeakers);
    R = F*Z*hlabels + sample_HTnoise(nu,rdim,N,W);
    
    
    [f,fi] = create_scalTrans();
    scal = pi;
    T = f(scal,R);
    
    ss = scal*exp(-1:0.1:1);
    y = zeros(size(ss));
    for i=1:length(ss)
        y(i) = testMLNDAobj(T,hlabels,F,W,fi,ss(i));
    end
    ystar = testMLNDAobj(T,hlabels,F,W,fi,scal);
    close all;
    semilogx(ss,y);hold;
    semilogx(scal,ystar,'*r');
    
end

function X = sample_HTnoise(nu,dim,n,W)
% Sample n heavy-tailed dim-dimensional variables. (Only for integer nu.)
%
% Inputs:
%   nu: integer nu >=1, degrees of freedom of resulting t-distribution
%   n: number of samples
%   W: precision matrix for T-distribution
%
% Output:
%   X: dim-by-n samples

    cholW = chol(W);    
    if isinf(nu)
        precisions = ones(1,n);
    else
        precisions = mean(randn(nu,n).^2,1);
    end
    std = 1./sqrt(precisions);
    X = cholW\bsxfun(@times,std,randn(dim,n));
end



    
