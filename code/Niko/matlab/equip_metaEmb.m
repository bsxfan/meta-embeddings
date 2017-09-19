function f = equip_metaEmb(f)
% Equip an existing meta-embedding implementation with some default derived
% functionality. Existing function handles are not overwritten. 
    
    if ~isfield(f,'llr')
        f.llr = @llr;
    end
    
    if ~isfield(f,'innerprod')
        f.innerprod = @innerprod;
    end
    
    if ~isfield(f,'norm_square')
        f.norm_square = @() f.innerprod(f);
    end
    
    if ~isfield(f,'expectation')
        f.expectation = @() exp(f.log_expectation());
    end
    
    if ~isfield(f,'scale')
        f.scale = @(s) f.shiftlogscale(log(s));
    end
    
    if ~isfield(f,'distance_square')
        f.distance_square = @(g) f.norm_square() + g.norm_square() - 2*f.innerprod(g);
    end
    
    if ~isfield(f,'norm_square_of_sum')
        f.norm_square_of_sum = @(g) f.norm_square() + g.norm_square() + 2*f.innerprod(g);
    end
    
    if ~isfield(f,'L1normalize')
        f.L1normalize = @() f.shiftlogscale(-f.log_expectation());
    end
    
    
    
    %computes y = log<fg> - log<f> - log<g>
    % optionally return some intermediate calculations
    function [y,fg,lognum,logden1,logden2] = llr(g) 
        fg = f.pool(g);

        lognum = fg.log_expectation();
        logden1 = f.log_expectation();
        logden2 = g.log_expectation(); 

        y = lognum - logden1 - logden2;
    end

    % returns y = <f,g> = <fg>
    function y = innerprod(g)
        fg = f.pool(g);
        y = fg.expectation();
    end




    







end