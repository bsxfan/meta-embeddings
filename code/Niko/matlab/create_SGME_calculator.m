function SGME = create_SGME_calculator(E)

    
    [V,D] = eig(E);  % E = VDV'
    d = diag(D);     % eigenvalues
    zdim = length(d);
    ii = reshape(logical(eye(zdim)),[],1);


    SGME.SGME2GME = @SGME2GME;
    SGME.log_expectations = @log_expectations;
    SGME.logLR = @logLR;
    SGME.plotAll = @plotAll;
    SGME.V = V;
    SGME.d = d;

    
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

    function y = log_expectations(A,b)
        bd = bsxfun(@times,b,d);
        logdets = sum(log1p(bd),1);
        Q = sum((A.^2)./(1+bd),1);
        y = (Q-logdets)/2;
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