function [y,Ezz,back] = posteriorNorm_mindiv_slow(A,B,b)
% Computes, for every i:  log N( 0 | Pi\A(:,i), inv(Pi) ), where
%
%   precisions are Pi = I + b(i)*B
%
% This is the slow version, used only to verify correctness of the function
% value and derivatives of the fast version, posteriorNorm_fast().
%
% Inputs:
%    A: dim-by-n, natural parameters (precision *mean) for n Gaussians    
%    B: dim-by-dim, common precision matrix factor (full, positive semi-definite)
%    b: 1-by-n, precision scale factors 
%
% Outputs:
%   y: 1-by-n, log densities, evaluated at zero
%   back: backpropagation handle, [dA,dB,db] = back(dy)

    if nargin==0
        test_this();
        return;
    end
    
    [dim,n] = size(A);
    I = speye(dim);
    
    y = zeros(1,n);
    S = zeros(dim,n);
    Ezz = zeros(dim);
    for i=1:n
        a = A(:,i);
        bBI = I+b(i)*B;
        s = bBI\a;
        S(:,i) = s;
        logd = logdet(bBI);
        y(i) = (logd - s.'*a)/2;
        Ezz = Ezz + inv(bBI) + s*s.';
    end
    Ezz = Ezz/n;

    back = @back_this;

    
    
    function [dA,dB,db] = back_this(dy,dEzz)
        dA = zeros(size(A));
        db = zeros(size(b));
        dB = zeros(size(B));
        
        for ii=1:n
            s = S(:,ii);
            a = A(:,ii);
            bBI = I+b(i)*B; 
            
            ds = (2/n)*(dEzz*s);
            
            da = -(dy(ii)/2)*s;
            ds = ds -(dy(ii)/2)*a;
            dlogd = dy(ii)/2;
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
    testBackprop(@posteriorNorm_slow,{A,B,b},{1,1,1});






end




