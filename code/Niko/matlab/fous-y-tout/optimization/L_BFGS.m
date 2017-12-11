function [w,y,mem,logs] = L_BFGS(obj,w,maxiters,timeout,mem,stpsz0,callback)
% L-BFGS Quasi-Newton unconstrained optimizer.
%   -- This has a small interface change from LBFGS.m --
%
% Inputs:
%   obj: optimization objective, with interface: [y,grad] = obj(w),
%        where w is the parameter vector, y is the scalar objective value 
%        and grad is a function handle, so that grad(1) gives the gradient 
%        (same size as w).
%   w: the initial parameter vector
%   maxiters: max number of LBFGS iterations (line search iterations do not 
%             count towards this limit).
%   timeout: LBFGS will stop when timeout (in seconds) is reached.
%   mem is either: (i) A struct with previously computed LBFGS data, to
%                      allow resumption of iteration.
%                  (ii) An integer: the size of the LBFGS memory. A good
%                       default is 20.
%   


    %some linesearch magic numbers
    maxfev = 20;  %max number of function evaluations
    stpmin = 1e-15; %same as Poblano default
    stpmax = 1e15; %same as Poblano default
    ftol = 1e-5; % as recommended by Nocedal (c1 in his book)
    gtol = 0.9; % as recommended by Nocedal (c2 in his book)
    xtol = 1e-15; %same as Poblano default
    quiet = false;

    %termination parameters
    %stopTol = 1e-5;  %same as Poblano
    relFuncTol = 1e-6; %same as Poblano
    
    if ~exist('stpsz0','var') || isempty(stpsz0)
        stpsz0 = 1;
    end
    stpsz = stpsz0;
    

    if ~exist('timeout','var') || isempty(timeout)
        timeout = 15*60;
        fprintf('timeout defaulted to 15 minutes');
    end;

    if ~exist('callback','var') || isempty(callback)
       ncbLogs = 0;
    else
       ncbLogs = length( callback(w) );
    end;

    tic;
    dim = length(w);

    if ~isstruct(mem)
        m = mem;
        mem = [];
        mem.m = m;
        mem.sz = 0;
        mem.rho = zeros(1,m);
        mem.S = zeros(dim,m);
        mem.Y = zeros(dim,m);
    else
        m = mem.m;
    end

    if ~exist('y','var') || isempty(y)
        [y,grad] = obj(w);
        g = grad(1);
        fprintf('LBFGS 0: obj = %g, ||g||=%g\n',y,sqrt(g'*g));
    end
    
    
    initial_y = y;
    
    logs = zeros(3+ncbLogs, maxiters);
    nlogs = 0;
    
    gmag = sqrt(g'*g);
    k = 0;
    while true

        if gmag< eps
            fprintf('LBFGS converged with tiny gradient\n');
            break;
        end
        
        
        % choose direction
        p = -Hprod(g,mem);
        assert(g'*p<0,'p is not downhill');
        
        % line search
        
        g0 = g;
        y0 = y;
        w0 = w;
        [w,y,grad,g,alpha,info,nfev] = minpack_cvsrch(obj,w,y,g,p,stpsz,...
                                        ftol,gtol,xtol, ...
                                        stpmin,stpmax,maxfev,quiet);        

        stpsz = 1;
                                    
        delta_total = abs(initial_y-y);
        delta = abs(y0-y);
        if delta_total>eps
            relfunc = delta/delta_total;
        else
            relfunc = delta;
        end
        
        gmag = sqrt(g'*g);
                                    
                                    
        if info==1 %Wolfe is happy
            sk = w-w0;
            yk = g-g0;
            dot = sk'*yk;
            assert(dot>0);
            if mem.sz==m
                mem.S(:,1:m-1) = mem.S(:,2:m);
                mem.Y(:,1:m-1) = mem.Y(:,2:m);
                mem.rho(:,1:m-1) = mem.rho(:,2:m);
            else
                mem.sz = mem.sz + 1;
            end
            sz = mem.sz;
            mem.S(:,sz) = sk;
            mem.Y(:,sz) = yk;
            mem.rho(sz) = 1/dot;
            fprintf('LBFGS %i: ||g||/n = %g, rel = %g\n',k+1,gmag/length(g),relfunc);
        else
            fprintf('LBFGS %i: NO UPDATE, info = %i, ||g||/n = %g, rel = %g\n',k+1,info,gmag/length(g),relfunc);
        end
        
        time = toc;
        nlogs = nlogs+1;
        if ncbLogs > 0
            logs(:,nlogs)  = [time; y; nfev; callback(w)'];
            disp(logs(4:end,nlogs)');
        else
            logs(:,nlogs)  = [time;y;nfev];
        end
        
        k = k + 1;
        if k>=maxiters
            fprintf('LBFGS stopped: maxiters exceeded\n');
            break;
        end
        
        if time>timeout
            fprintf('LBFGS stopped: timeout\n');
            break;
        end

        if relfunc < relFuncTol
            fprintf('\nTDN: stopped with minimal function change\n');
            break;
        end
        
    end
    logs = logs(:,1:nlogs);    

end



function r = Hprod(q,mem)
    if mem.sz==0
        r = q;
        return;
    end
    sz = mem.sz;
    S = mem.S;
    Y = mem.Y;
    rho = mem.rho;
    alpha = zeros(1,sz);
    for i=sz:-1:1
        alpha(i) = rho(i)*S(:,i)'*q;
        q = q - alpha(i)*Y(:,i);
    end
    yy = sum(Y(:,sz).^2,1);
    r = q/(rho(sz)*yy);
    for i=1:sz
        beta = rho(i)*Y(:,i)'*r;
        r = r + S(:,i)*(alpha(i)-beta);
    end
end
