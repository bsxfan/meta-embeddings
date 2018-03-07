function be = create_HTPLDA_SGME_backend(nu,F,W)
% Constructs a fast, approximate HT-PLDA backend for i-vector scoring.  
%
% The generative model fantasy goes as follows: 
% - The speaker identity variable, z \in R^zdim, is sampled (once) 
% independently for every speaker, from the, standard multivariate normal 
% distribution of dimensionality zdim. 
% - For a speaker represented by z, every new i-vector, r \in R^D, is 
% sampled from the t-distribution, T(F*z, nu, W), where nu is degrees of 
% freedom and W is within-speaker precision.
%
% For sdmall nu, this model has D-dimensional, heavy-tailed 'channel' noise 
% (t-distribution with degrees of freedom nu). The speaker identity 
% likelihoods, P(r | z), as a function of z, are proportional to t-distributions
% in z, but, for D >> d are almost Gaussian, with increased degrees of 
% freedom = nu+D-d. Approximating these likelihoods as Gaussians gives our
% shortcut, which gives fast scoring.
%
% The backend extracts an intermediate representation, termed 'meta-embedding' 
% from every i-vector. The meta-embeddings are Gaussian approximations to the 
% speaker identity likelihood functions. Meta-embeddings are represented by
% the Gaussian natural parameters. All precisions are mutually 
% diagonalizable (they differ only by a scale factor), giving further efficiency.
%
% - Meta-embeddings from multiple enrollment meta-embeddings can be pooled
%   if required. Pooling is just multiplication of the likelihood functions---
%   in practice, addition of natural parameters).
% - Raw, or pooled meta-embeddings from test and enrollment i-vectors can be  
%   scored. These scores are (approximate) log-likelihood-ratios.
%
% Inputs (HT-PLDA model parameters:
%   nu: scalar > 0, degrees of freedom
%   F: D-by-d speaker factor loading matrix, D >> d, where D is i-vector
%      dimension and d is speaker factor dimension.
%   W: D-by-D, positive definite within-class precision
%
% Output:
% 
%  be: the backend, represented by a struct, which acts like a 'home-made' 
%      object that encapsulates the model parameters as local variables. 
%      Various scoring 'methods' are made available as fields of this struct, in 
%      the form of function handles.  
% 
%      For documentation on these methods, see the m-file source of this
%      function.
%
% Typical simple usage:
%
%   > be = HTPLDA_SGME_train_VB(Training,labels,nu,zdim,niters); 
%   > scores = be.score_trials(Enroll,[],Test);
%
%   - HTPLDA_SGME_train_VB invokes create_HTPLDA_SGME_backend.
%   - Training, Enroll and Test are matrices of i-vectors.
%
%
% Advanced usage allows: 
% - (by-the-book) pooling of multiple enrollment i-vectors per target speaker
% - storage of the intermediate meta-embeddings, which are more compact
%   than i-vectors
% - very generally, scoring of likelihood-ratios between any propositions that
%   can be represented as partitions w.r.t. speaker of a set of i-vectors.




    be.getParams = @ getParams;
    be.extract = @extract;
    be.log_expectations = @log_expectations;
    be.normalize = @normalize;
    be.pool = @pool;
    be.log_inner_products = @inner_products;
    be.score_trials = @score_trials;
    be.enroll  = @enroll;

    [D,d] = size(F);

    P = F.'*W;         % d-by-D
    B0 = P*F;          % d-by-d common precision (up to scaling)
    [V,L] = eig(B0);   % eigendecomposition B0 = V*L*V'; L is diagonal, V is orthonormal  
    L = diag(L);
    VP = V.'*P;
    
    G = W - VP.'*bsxfun(@ldivide,L,VP);   % inv(B0) = V*inv(L)*V'

    
    
    function [nu1,F1,W1] = getParams()             %OK
        [nu1,F1,W1] = deal(nu,F,W);
    end
    
    
    function meta_embeddings = extract(R,do_norm)  %OK
        q = sum(R.*(G*R),1);
        b = (nu+D-d)./(nu+q);  
        A = bsxfun(@times,b,VP*R); 
        meta_embeddings.A = A;
        meta_embeddings.b = b;
        if ~exist('do_norm','var') || isempty(do_norm) || do_norm
            meta_embeddings = normalize(meta_embeddings);
        end
        meta_embeddings.L = L;
    end


    function e = log_expectations(meta_embeddings)  %OK
        A = meta_embeddings.A;
        b = meta_embeddings.b;

        bL1 = 1 + bsxfun(@times,b,L);
        logdets = log(prod(bL1,1));
        Q = sum(A.^2./bL1,1);    
        e = (Q-logdets)/2;
        if isfield(meta_embeddings,'logscal')
            e = e + meta_embeddings.logscal;
        end
    end


    function meta_embeddings = normalize(meta_embeddings)   %OK
        if ~isfield(meta_embeddings,'logscal')
            meta_embeddings.logscal = -log_expectations(meta_embeddings);
        end
    end


    function pooled_meta_embeddings = pool(meta_embeddings,Flags,do_norm)
        if ~exist('Flags','var') || isempty(Flags)
            pooled_meta_embeddings = meta_embeddings;
        else
            pooled_meta_embeddings.A = meta_embeddings.A * Flags.';
            pooled_meta_embeddings.b = meta_embeddings.b * Flags.';
        end
        if ~exist('do_norm','var') || isempty(do_norm) || do_norm
            pooled_meta_embeddings = normalize(pooled_meta_embeddings);
        end
        
    end


    function X = log_inner_products(Left,Right)
        B = bsxfun(@plus,Left.b.',Right.b);
        [m,n] = size(B);
        X = zeros(m,n);
        for j=1:n
            AA = bsxfun(@plus,Left.A,Right.A(:,j));
            me.A = AA;
            me.b = B(:,j).';
            X(:,j) = log_expectations(me).';
        end
        sl = isfield(Left,'logscal');
        sr = isfield(Right,'logscal');
        if sl&&sr
            X = X + bsxfun(@plus,Left.logscal.',Right.logscal);
        elseif sl && ~sr
            X = bsxfun(@plus,X,Left.logscal.');
        elseif ~sl && sr
            X = bsxfun(@plus,X,Right.logscal);
        end
    end

    function target_meta_embeddings = enroll(R,Target_flags)
        target_meta_embeddings = pool(extract(R,false),Target_flags,true);
    end


    function logLRs = score_trials(Enroll,Target_flags,Test)
        Left = enroll(Enroll,Target_flags);
        Right = extract(Test);
        logLRs = log_inner_products(Left,Right);
    end

    
end