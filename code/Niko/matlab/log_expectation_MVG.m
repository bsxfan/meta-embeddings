function y = log_expectation_MVG(a,B)
% f(z) =  exp[a'z - z'Bz/2]
% pi(z) = N(z | 0, I)
% y = log E{f}_pi
%
% note: a,B are the natural parameters for the unnormalized Gaussian f(z)
% 
% inputs: 
%   a: d-vector
%   B: struct representing d-by-d pos semi-def matrix
%   B.BI: struct representing pos def matrix I+B
%   B.BI.solve: returns mean and log det on input a

    [mu,log_det] = B.BI.solve(a);
    y = (mu'*a + log_det)/2;
    
end