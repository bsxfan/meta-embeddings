function C = create_diagonalized_C(B,R,RM,Ra,W,M,a)
%  Creates object to represent: C = inv(lambda W + B),   
%
%  Inputs:
%    B: positive definite matrix (i-vector dimension)
%    R: chol(W), so that R'R=W  (i-vector dimension)
%    RM: R*M, where M has language means in columns
%    Ra: (R')\a, vector (i-vector dimension)
%    W,M,a: optional for verification with slow version

    if nargin==0
        test_this();
        return;
    end

    dim = length(Ra);
    

    K = (R.'\B)/R;
    [V,D] = eig(K);          %K = V*D*V'
    e = diag(D);             %eigenvalues
    
    mWm = sum(RM.^2,1);  %vector of m'Wm, for every language
    VRM = V.'*RM;
    VRa = V.'*Ra;
    
    VRaVRa = VRa.^2;
    VRMVRM = VRM.^2;
    VRMVRa = bsxfun(@times,VRa,VRM);
    
    
    
    C.traceCW = @traceCW;
    C.logdetCW = @logdetCW;
    C.quad = @quad;

    C.slowQuad = @slowQuad;
    C.slow_traceCW = @slow_traceCW;
    C.slow_logdetCW = @slow_logdetCW;
    
    C.lambda_by_root = @lambda_by_root;
    C.lambda_by_min = @lambda_by_min;
    
    %C.slow_xCWCx = @slow_xCWCx;
    %C.xCWCx = @xCWCx;
    %C.xCx = @xCx;
    %C.slow_xCx = @slow_xCx;
    
    
    function lambda = lambda_by_root(nu,log_lambda,ell)
        f = @(log_lambda) log_lambda - log((nu+dim)/(nu+energy(log_lambda,ell)));
        lambda = fzero(f,log_lambda);
    end
    
    
    function lambda = lambda_by_min(nu,log_lambda)
        f = @(log_lambda) (log_lambda - log((nu+dim)/(nu+energy(log_lambda))))^2;
        lambda = fminsearch(f,log_lambda);          
    end

    
    function y = energy(log_lambda,ell)
        lambda = exp(log_lambda);
        y = quad(lambda) + traceCW(lambda);
        y = y(ell);
    end

    
    function y = quad(lambda)
        s = lambda + e;
        ss = s.^2;
        mWmu = sum(bsxfun(@rdivide, lambda*VRMVRM + VRMVRa, s),1);
        muWmu = sum(bsxfun(@rdivide,bsxfun(@plus,lambda^2*VRMVRM + (2*lambda)*VRMVRa, VRaVRa), ss),1);
        y = mWm + muWmu -2*mWmu;
    end
    


    function y = slowQuad(lambda)
        P = lambda*W + B;
        cholP = chol(P);
        Mu = cholP\(cholP'\bsxfun(@plus,lambda*W*M,a));
        delta = R*(Mu-M);
        y = sum(delta.^2,1);
        %y = sum(Mu.*(W*M),1);
    end

%     function y = xCx(lambda,x)
%         z = V'*((R.')\x);
%         s = lambda + e;
%         y = sum(z.^2./s,1);
%     end
% 
%     function y = xCWCx(lambda,x)
%         z = V'*((R.')\x);
%         s = lambda + e;
%         y = sum(z.^2./s.^2,1);
%     end
% 
%     function y = slow_xCx(lambda,x)
%         P = lambda*W+B;
%         y = x'*(P\x);
%     end
%     
%     function y = slow_xCWCx(lambda,x)
%         P = lambda*W+B;
%         z = P\x;
%         y = z.'*W*z;
%     end

    
    function [y,back] = traceCW(lambda)
        s = lambda + e;
        r = 1./s;
        y = sum(r,1);
        back = @back_this;
        function dlambda = back_this(dy)
            dr = dy;
            ds = (-dr)*r./s;
            dlambda = sum(ds,1);
        end
    end

    function y = slow_traceCW(lambda)
        P = lambda*W + B;
        cholP = chol(P);
        X = cholP.'\R.';
        y = X(:).'*X(:);
    end

    function [y,back] = logdetCW(lambda)
        s = log(lambda) + log1p(e/lambda);
        y = -sum(s,1);
        back = @(dy) (-dy)*sum(1./(lambda+e));
    end

    function y = slow_logdetCW(lambda)
        P = lambda*W + B;
        cholP = chol(P);
        y = 2*( sum(log(diag(R))) - sum(log(diag(cholP))) );
    end

end

function test_this()

    dim = 400;
    L = 1;
    RR = randn(dim,dim);W = RR*RR';
    RR = randn(dim,dim);B = RR*RR';
    a = randn(dim,1);
    M = randn(dim,L);
    R = chol(W);
    
    C = create_diagonalized_C(B,R,R*M,(R')\a,W,M,a);
    
    lambda = rand/rand;
    
    %x = randn(dim,1);
    %[C.xCx(lambda,x),C.slow_xCx(lambda,x)]
    
    %tic;C.quad(lambda);toc
    %tic;C.slowQuad(lambda);toc

    %C.quad(lambda)
    %C.slowQuad(lambda)
    
    %[C.traceCW(lambda),C.slow_traceCW(lambda)]
    %[C.logdetCW(lambda),C.slow_logdetCW(lambda)]
    
    %[C.xCWCx(lambda,x),C.slow_xCWCx(lambda,x)]
    

    C.lambda_by_root(1,1)
    C.lambda_by_root(1,10)
    C.lambda_by_root(1,0.1)
    
    C.lambda_by_min(1,1)
    C.lambda_by_min(1,10)
    C.lambda_by_min(1,0.1)
    
    a = a*0;
    B = B*0;
    C = create_diagonalized_C(B,R,R*M,(R')\a,W,M,a);
    
    C.lambda_by_root(0.1,0.01)
    C.lambda_by_root(1,10)
    C.lambda_by_root(10,0.1)
    
    C.lambda_by_min(1,10)
end

