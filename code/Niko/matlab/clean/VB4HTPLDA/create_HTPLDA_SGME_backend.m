function be = create_HTPLDA_SGME_backend(nu,F,W)
% Constructs a fast, approximate HT-PLDA backend for i-vector scoring.  
%
% The generative model fantasy goes as follows: 
% - The speaker identity variable, z \in R^d, is sampled (once) 
%   independently for every speaker, from the standard Gaussian of dimensionality zdim. 
% - For a speaker represented by z, every new i-vector, r \in R^D, is 
%   sampled from the t-distribution, T(F*z, nu, W), where nu is degrees of 
%   freedom; F (D-by-d) is the factor loading matrix; and W is within-speaker 
%   precision.
%
% For small nu, this model has D-dimensional, heavy-tailed 'channel noise'. 
% The speaker identity likelihoods, P(r | z) as a function of z, are 
% proportional to t-distributions in z. But for D >> d these t-distytibutions
% are almost Gaussian, with increased degrees of freedom = nu+D-d. 
% Approximating these likelihoods as Gaussians gives our shortcut, which 
% gives fast scoring.
%
% The backend extracts an intermediate representation, termed 'meta-embedding' 
% from every i-vector. The meta-embeddings are Gaussian approximations to the 
% speaker identity likelihood functions. Meta-embeddings are represented by
% the Gaussian natural parameters. All precisions are mutually 
% diagonalizable (they differ only by a scale factor), giving further efficiency.
% SGME is for simple Gaussian meta-embedding.
%
% - Meta-embeddings extracted from multiple enrollment i-vectors can be pooled
%   if required. Pooling is just multiplication of the likelihood
%   functions (in practice addition of natural parameters).
% - Raw or pooled meta-embeddings from test and enrollment i-vectors can be  
%   scored. These scores are (approximate) log-likelihood-ratios.
% - Meta-embeddings can be normalized, which is an optional additional step
%   that can be done after extraction, or pooling. Pooling destroys 
%   normalization, whereupon normalization has to be re-done. Inner products 
%   between normalized meta-embeddings give likelihood-ratios. 
%  
%
% Inputs (HT-PLDA model parameters:
%   nu: scalar > 0, degrees of freedom
%   F: D-by-d speaker factor loading matrix, D >> d, where D is i-vector
%      dimension and d is speaker factor dimension.
%   W: D-by-D, positive definite within-class precision
%
% Outputs:
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
% - (By-the-book) pooling of multiple enrollment i-vectors per target
%   speaker.
% - Representation of i-vectors by meta-embeddings, which are more compact
%   than i-vectors.
% - Very generally, scoring of likelihood-ratios between any propositions that
%   can be represented as partitions (w.r.t. speaker) of a set of i-vectors.



    % list of methods --- see method definitions below for details 
    be.getParams = @ getParams;
    be.extract = @extract;
    be.log_expectations = @log_expectations;
    be.normalize = @normalize;
    be.pool = @pool;
    be.log_inner_products = @inner_products;
    be.score_trials = @score_trials;
    be.enroll  = @enroll;

    
    
    % precomputation 
    [D,d] = size(F);

    P = F.'*W;         % d-by-D
    B0 = P*F;          % d-by-d common precision (up to scaling)
    [V,L] = eig(B0);   % eigendecomposition B0 = V*L*V'; L is diagonal, V is orthonormal  
    L = diag(L);
    VP = V.'*P;
    
    G = W - VP.'*bsxfun(@ldivide,L,VP);   % inv(B0) = V*inv(L)*V'

    
    
%%%%%%   Method definitions %%%%%%%    
    
    function [nu1,F1,W1] = getParams()             
    % Returns the parameters of the generative HT-PLDA model. See details 
    % model and parameters above.    
        [nu1,F1,W1] = deal(nu,F,W);
    end
    
    
    function meta_embeddings = extract(R,do_norm)
    % Extracts the intermediate representations (meta-embeddings) for all of the i-vectors
    % in the input R. All scoring can be done as functions of
    % meta-embeddings.
    %
    % Inputs:
    %   R: D-by-N matrix of i-vectors (N of them)
    %   do_norm: [optional, default = true] logical flag to request 
    %            normalization of the meta-embeddings.
    % 
    % Output:
    %   meta-embeddings: A struct containing the N meta-embeddings
    %                    extracted from the N inpputs i-vectors.
    %     meta_embedings.A: D-by-N natural parameters (precision * mean)
    %     meta_embedings.B: 1-by-N natrual parameters (precision scaling factors)
    %     meta_embedings.logscal: 1-by-N (optional, if normalized)
    %     meta_embeddings.L: d-by-1 eigenvectors of common precision 
    %                        (currently not used in any other methods)
    
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


    function e = log_expectations(meta_embeddings)  
    % The (log) expected value of the meta-embedding likelihood function
    % w.r.t. the standard normal prior. This is a fundamental building
    % block in all scoring operations. For meta-embeddings representing
    % singleton or pooled sets of i-vectors, this return the log-likelihood
    % that the set of i-vectors belong to a common speaker.
    %
    % Input: meta-embeddings: struct for N meta-embeddings
    % Output: e: 1-by-N: log-expectations for the input meta-embeddings
    
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


    function meta_embeddings = normalize(meta_embeddings)   
    % This normalizes the input meta-embeddings (which are functions) by 
    % dividing by the expected values of the functions (which are scalars).
    % Input: meta_embeddings
    % Output: meta_embeddings (normalized)
    
        if ~isfield(meta_embeddings,'logscal')
            meta_embeddings.logscal = -log_expectations(meta_embeddings);
        end
    end


    function pooled_meta_embeddings = pool(meta_embeddings,Flags,do_norm)
    % Given N meta-embeddings, outputs M (possibly) pooled meta-embeddings.
    % Pooling is typically used for multi-enroll trials, but can be used to
    % assemble more complex LR scores.
    %
    % Inputs:
    %   meta_embeddings: struct containing N of them
    %   Flags: M-by-N logical matrix, each row identifies a subset to be pooled
    %          optional: can be [], in which case identity is implied: outputs = inputs.
    %   do_norm: [optional, default = true] logical flag to request 
    %            normalization of the outputs after pooling. (Pooling
    %            destroys normalization
    %
    % Output: pooled_meta_embeddings:  struct containing M of them
    
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
    % Computes m-by-n matrix of (log) inner products between each of 
    % m meta-embeddings with each of n meta-embeddings. The inner product
    % between two meta-embeddings (which are functions) is the expected
    % value w.r.t. the standard normal pof the product of the two
    % functions.
    %
    % This is useful for scoring binary verification trials.
    % (Multi-enrollment is done by pooling of one of the inpout arguments.)
    % 
    % Inner products could be calculated via pooling and expectation, but this
    % method does the calculation conveniently and efficiently for the whole matrix.
    %
    % Inputs:
    %   Left: struct containing m of them
    %   Right: struct containing n of them
    %
    % Output: X: m-by-n matrix of log inner products
    % 
    % Note: if the inputs have all been normalized, then the inner products are also
    % likelihood ratios.
     
        
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
    % Convenience method to extract, pool and normalize the meta-embeddings 
    % to represent a number of target speakers.
    %
    % Inputs:
    %   R: D-by-N matrix of i-vectors (N of them)
    %   Target_flags: M-by-N logical matrix, the rows of which indicate the
    %                 subsets of i-vectors to be pooled for each of M
    %                 target speakers.
    %                 optional: can be [], in which case identity is
    %                           assumed
    %   Output: target_meta_embeddings: struct containing M of them
    
    
        target_meta_embeddings = pool(extract(R,false),Target_flags,true);
    end


    function logLRs = score_trials(Enroll,Target_flags,Test)
    % Convenience method to score a whole evaluation database of i-vectors.
    % Inputs:
    %   Enroll: D-by-K matrix of enrollment i-vectors (K of them)
    %   Target_flags: K-by-M logical matrix. See definition for the same 
    %                 parameter in enroll() above.
    %   Test: D-by-N matrix of test i-vectors (N of them)
    %   Output: logLRs: M-by-N matrix of log LR scores.

    
        Left = enroll(Enroll,Target_flags);
        Right = extract(Test);
        logLRs = log_inner_products(Left,Right);
    end

    
end