function model = create_semi_discrete_plda3(N,dim,scal)

    if nargin==0
        test_this();
        return;
    end

    prior = -log(N);  %flat prior on speaker identity variable
    Means = randn(dim,N); 
    W = randn(dim,dim+2);
    W = W*W.'*(scal/(dim+2));
    
    cholW = chol(W);
    WMeans = W*Means;
    offs = sum(Means.*WMeans,1)/2;
    F = [WMeans.',-offs.'];
    
    model.sample = @sample;
    model.extract_me = @extract_me;
    model.log_expectations = @log_expectations;
    
    
    function [D,Z] = sample(HL,Z)
    % HL: K-by-T, label matrix, with 1-hot columns, for T samples from K speakers
    
        [K,T] = size(HL);

        if ~exist('Z','var') || isempty(Z)
            %sample Z from flat prior
            Z = sparse(randi(N,1,K),1:K,true,N,K);  % N-by-K: speaker identity variables (1-hot columns)
        end

        %generate data
        MZ = Means*Z;  % dim-by-K
        D = MZ*HL + cholW\randn(dim,T); 
    end



    function E = extract_me(D)
        %E = bsxfun(@minus,WMeans'*D,offs.');
        E = [D;ones(1,size(D,2))];
    end


    function L = log_expectations(E)
        E = F*E;
        mx = max(E,[],1);
        L = prior + mx + log(sum(exp(bsxfun(@minus,E,mx)),1));  
    end





end


function test_this()

    close all;

    N = 1000;
    dim = 100;
    scal = 0.2;
    model = create_semi_discrete_plda3(N,dim,scal);
    llhfun = @model.log_expectations;
    extr = @model.extract_me;

    n = 10000;
    HL = logical(speye(n));
    [Enroll,Z] = model.sample(HL);
    Enroll = extr(Enroll);
    Tar = extr(model.sample(HL,Z));
    Non = extr(model.sample(HL));
    
    llr = @(enr,test) llhfun(enr + test) - llhfun(enr) - llhfun(test);
    
    tar = llr(Enroll,Tar);
    non = llr(Enroll,Non);
    
    fprintf('EER = %g%%\n',100*eer(tar,non));
    fprintf('Cllr,minCllr = %g, %g\n',cllr(tar,non),min_cllr(tar,non));
    
    subplot(1,3,1);hist([tar,non],300);
    subplot(1,3,2);hist(non,300);
    subplot(1,3,3);hist(tar,300);
    
    plot_type = Det_Plot.make_plot_window_from_string('old');
    plot_obj = Det_Plot(plot_type,'SEMI-DISCRETE-PLDA');

    plot_obj.set_system(tar,non,'sys1');
    plot_obj.plot_steppy_det({'b','LineWidth',2},' ');
    
    
end

