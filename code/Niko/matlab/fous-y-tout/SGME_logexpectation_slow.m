function [y,back] = SGME_logexpectation_slow(A,b,B)
% log expected values (w.r.t. standard normal) of diagonalized SGMEs
% Inputs:
%    A: dim-by-n, natural parameters (precision *mean) for n SGMEs    
%    b: 1-by-n, precision scale factors for these SGMEs
%    B: dim-by-dim, common precision (full) matrix factor 
%
% Note:
%    A(:,j) , b(j)*B forms the meta-embedding for case j
%
% Outputs:
%   y: 1-by-n, log expectations
%   back: backpropagation handle, [dA,db,dB] = back(dy)


    if nargin==0
        test_this();
        return;
    end
    
    [dim,n] = size(A);
    I = speye(dim);
    
    y = zeros(1,n);
    S = zeros(dim,n);
    for i=1:n
        a = A(:,i);
        bBI = I+b(i)*B;
        s = bBI\a;
        S(:,i) = s;
        logd = logdet(bBI);
        y(i) = (s.'*a - logd)/2;
    end

    back = @back_this;

    
    
    function [dA,db,dB] = back_this(dy)
        dA = zeros(size(A));
        db = zeros(size(b));
        dB = zeros(size(B));
        
        for ii=1:n
            s = S(:,ii);
            a = A(:,ii);
            da = (dy(ii)/2)*s;
            ds = (dy(ii)/2)*a;
            dlogd = -dy(ii)/2;
            bBI = I+b(ii)*B;
            dbBI = dlogd*inv(bBI); %#ok<MINV>
            da2 = bBI.'\ds;
            dA(:,ii) = da + da2;
            dbBI = dbBI - (da2)*s.';
            dB = dB + b(ii)*dbBI;
            db(ii) = dbBI(:).'*B(:);
        end
    end


end

function y = logdet(M)
    [~,U] = lu(M);
    y = sum(log(diag(U).^2))/2;
end


function test_this()

    m = 3;
    n = 5;

    A = randn(m,n);
    b = rand(1,n);
    B = randn(m,m+1); B = B*B.';
    testBackprop(@SGME_logexpectation_slow,{A,b,B},{1,1,1});






end




