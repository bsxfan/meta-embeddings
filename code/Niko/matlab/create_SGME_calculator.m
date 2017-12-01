function [SGME,LEfun] = create_SGME_calculator(E)

    if nargin==0
        test_this();
        return;
    end

    
    [V,D] = eig(E);  % E = VDV'
    d = diag(D);     % eigenvalues
    dd = zeros(size(d)); %gradient w.r.t. d backpropagated from log_expectations
    zdim = length(d);
    ii = reshape(logical(eye(zdim)),[],1);


    SGME.SGME2GME = @SGME2GME;
    SGME.log_expectations = @log_expectations;
    SGME.logLR = @logLR;
    SGME.plotAll = @plotAll;
    SGME.V = V;
    SGME.d = d;
    LEfun = @LE;

    
    function reset_parameter_gradient()
        dd(:) = 0;
    end
    
    function dd1 = get_parameter_gradient()    
        dd1 = dd;
    end


    function plotAll(A,b,matlab_colours, tikz_colours, rotate)
        if ~exist('rotate','var') || isempty(rotate)
            rotate = true;
        end
        if ~exist('tikz_colours','var')
            tikz_colours = [];
        end
        [A,B] = SGME2GME(A,b,rotate);
        n = length(b);
        for i=1:n
            Bi = reshape(B(:,i),zdim,zdim);
            mu = Bi\A(:,i);
            if ~isempty(tikz_colours)
                plotGaussian(mu,inv(Bi),tikz_colours{i},matlab_colours{i});
            else
                plotGaussian(mu,inv(Bi),[],matlab_colours{i});
            end
        end
        
    end
    
    
    function [A,B] = SGME2GME(A,b,rotate)
        B = zeros(zdim*zdim,length(b));
        B(ii,:) = bsxfun(@times,b,d);
        if ~exist('rotate','var') || isempty(rotate) || rotate  %rotate by default
            A = V*A;
            for j = 1:size(B,2)
                BR = V*reshape(B(:,j),zdim,zdim)*V.';
                B(:,j) = BR(:);
            end
        end
    end

    function [y,back] = log_expectations(A,b)
        [y,back0] = LE(A,b,d);
        back = @back_this;
        function [dA,db] = back_this(dy)
            [dA,db,dd0] = back0(dy);
            dd = dd + dd0;
        end
    end


    function Y = logLR(left,right)
        B = bsxfun(@plus,left.b.',right.b);
        [m,n] = size(B);
        Y = zeros(m,n);
        for i=1:m
            AA = bsxfun(@plus,left.A(:,i),right.A);
            Y(i,:) = log_expectations(AA,B(i,:));
        end
    end
    
end


function [y,back] = LE(A,b,d)
    bd = bsxfun(@times,b,d);
    logdets = sum(log1p(bd),1);
    den = 1 + bd;
    Aden = A./den;
    Q = sum(A.*Aden,1);    %Q = sum((A.^2)./den,1);
    y = (Q-logdets)/2;

    back = @back_LE;


    function [dA,db,dd] = back_LE(dy)
        dQ = dy/2;
        %dlogdets = - dQ;

        dAden = bsxfun(@times,dQ,A);           
        dA = bsxfun(@times,dQ,Aden);           

        dA2 = dAden./den;
        dA = dA + dA2;          
        dden = -Aden.*dA2;

        dbd = dden - bsxfun(@rdivide,dQ,den);    %dlogdets = -dQ

        db = d.' * dbd;
        dd = dbd * b.';
    end


end






function test_this()

    m = 3;
    n = 5;
    A = randn(m,n);
    b = rand(1,n);
    d = rand(m,1);
    
    testBackprop(@LE,{A,b,d},{1,1,1});

end

