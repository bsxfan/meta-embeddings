function prior = create_uvg_Prior(mu,v)

    prior.sample = @sample;
    prior.logPDF = @logPDF;
    prior.Hessian = @logPDF_Hessian;

    function x = sample(n)
        if nargin==0
            n = 1;
        end
        x = mu+sqrt(v)*randn(1,n);
    end

    % y is (unnormalized) logPDF
    % y1 is dy/dx
    % y2 is dy1/dx
    function [y,y1] = logPDF(x)
        delta = x-mu;
        y = -delta.^2/(2*v);
        if nargout>=2
            y1 = -delta/v;
        end
    end


    function h = logPDF_Hessian(x)
        h = -1/v;
    end




end