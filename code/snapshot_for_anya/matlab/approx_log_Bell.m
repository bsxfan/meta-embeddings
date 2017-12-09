function [logB,err] = approx_log_Bell(n)

    logn = log(n);
    loglogn = log(logn);
    
    logB = n.*( logn -loglogn -1 + (1+loglogn)./logn + 0.5*(loglogn./logn).^2 );
    err = n.*loglogn./logn.^2;
end