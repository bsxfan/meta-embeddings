function model = create_semi_discrete_plda2(N,dim,scal)

    if nargin==0
        test_this();
        return;
    end

    prior = -2*log(N);  %flat prior on speaker identity variable
    Means = randn(dim,N); 
    W = randn(dim,dim+2);
    W = W*W.'*(scal/(dim+2));
    
    cholW = chol(W);
    WMeans = W*Means;
    offs = -sum(Means.*WMeans,1)/2;
    
    thr = -30; %-2*log(N);    % to controll sparsity
    
    B0 = -Means.'*WMeans;
    b = max(B0(:));
    B0 = B0-b;
    B1 = B0;
    B1(B1<thr) = -inf;
    B1 = sparse(exp(B1));
    B1 = exp(B1);
    B{1} = B1;
    
    B1 = 2*B0;
    B1(B1<thr) = -inf;
    B1 = sparse(exp(B1));
    B{2} = B1;
    
    
    F = [WMeans.',offs.'];
    
    model.sample = @sample;
    model.extract_me = @extract_me;
    model.log_expectations = @log_expectations;
    
    
    function [D,Z1,Z2] = sample(HL,Z1,Z2)
    % HL: K-by-T, label matrix, with 1-hot columns, for T samples from K speakers
    
        [K,T] = size(HL);

        if ~exist('Z1','var') || isempty(Z1)
            %sample Z from flat prior
            Z1 = sparse(randi(N,1,K),1:K,true);  % N-by-K: speaker identity variables (1-hot columns)
            Z2 = sparse(randi(N,1,K),1:K,true);  % N-by-K: speaker identity variables (1-hot columns)
        end

        %generate data
        MZ1 = Means*Z1;  % dim-by-K
        MZ2 = Means*Z2;  % dim-by-K
        D = (MZ1+MZ2)*HL + cholW\randn(dim,T); 
    end



    function E = extract_me(D)
        T = size(D,2);
        E = [D;ones(1,T)];
    end


    function L = log_expectations(E)
        V = F*E;
        mx = max(V,[],1);
        V = bsxfun(@minus,V,mx);
        V(V<thr) = -inf;
        V = exp(V);

        n = E(end,:);
        T = length(n);
        L = zeros(1,T);
        for i=1:max(n)
            f = n==i;
            Vi = V(:,f);
            L(f) = prior + i*b + 2*mx(f) + log(sum(Vi.*(B{i}*Vi),1));  
        end
    end





end


function test_this()

    N = 1000;
    dim = 100;
    scal = 0.1;
    model = create_semi_discrete_plda2(N,dim,scal);
    llhfun = @model.log_expectations;
    extr = @model.extract_me;

    n = 10000;
    HL = logical(speye(n));
    [Enroll,Z1,Z2] = model.sample(HL);
    Enroll = extr(Enroll);
    Tar = extr(model.sample(HL,Z1,Z2));
    Non = extr(model.sample(HL));
    
    llr = @(enr,test) llhfun(enr + test) - llhfun(enr) - llhfun(test);
    
    tar = llr(Enroll,Tar);
    non = llr(Enroll,Non);
    
    fprintf('EER = %g%%\n',100*eer(tar,non));
%     fprintf('Cllr,minCllr = %g, %g\n',cllr(tar,non),min_cllr(tar,non));
%     
%     hist([tar,non],300);
%     
%     plot_type = Det_Plot.make_plot_window_from_string('old');
%     plot_obj = Det_Plot(plot_type,'SEMI-DISCRETE-PLDA');
% 
%     plot_obj.set_system(tar,non,'sys1');
%     plot_obj.plot_steppy_det({'b','LineWidth',2},' ');
    
    
end

