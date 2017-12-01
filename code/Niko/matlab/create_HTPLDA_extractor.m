function HTPLDA = create_HTPLDA_extractor(F,nu,W)

    if nargin==0
        test_PsL();
        %test_this();
        return;
    end

    [rdim,zdim] = size(F);
    assert(rdim>zdim);
    nu_prime = nu + rdim - zdim;
    
    if ~exist('W','var') || isempty(W)
        W = speye(rdim);
    end
    
    E = F.'*W*F;
    G = W - W*F*(E\F.')*W;

    SGME = create_SGME_calculator(E);
    
    V = SGME.V;  % E = VDV'
    VFW = V.'*F.'*W;
    
    HTPLDA.extractSGMEs = @extractSGMEs;
    HTPLDA.SGME = SGME;
    HTPLDA.plot_database = @plot_database;
    
    
    function [A,b] = extractSGMEs(R)
        q = sum(R.*(G*R),1);
        b = nu_prime./(nu+q);
        A = bsxfun(@times,b,VFW*R);
    end
    
    matlab_colours = {'r','g','b','m','c','k',':r',':g',':b',':m',':c',':k'}; 
    tikz_colours = {'red','green','blue','magenta','cyan','black','red, dotted','green, dotted','blue, dotted','magenta, dotted','cyan, dotted','black, dotted'}; 


    function plot_database(R,labels,Z)
        assert(max(labels) <= length(matlab_colours),'not enough colours to plot all speakers');
        [A,b] = extractSGMEs(R);
        %SGME.plotAll(A,b,matlab_colours(labels), tikz_colours(labels));
        SGME.plotAll(A,b,matlab_colours(labels), []);
        if exist('Z','var') && ~isempty(Z)
            for i=1:size(Z,2)
                plot(Z(1,i),Z(2,i),[matlab_colours{i},'*']);
            end
        end
            
    end
    

end

function test_this()

    zdim = 2;
    xdim = 20;      %required: xdim > zdim
    nu = 3;         %required: nu >= 1, integer, DF
    fscal = 3;      %increase fscal to move speakers apart
    
    F = randn(xdim,zdim)*fscal;

    
    HTPLDA = create_HTPLDA_extractor(F,nu);
    SGME = HTPLDA.SGME;
    
    %labels = [1,2,2];
    %[R,Z,precisions] = sample_HTPLDA_database(nu,F,labels);
    
    
    n = 8;
    m = 5;
    %prior = create_PYCRP(0,[],m,n);
    prior = create_PYCRP([],0,m,n);
    [R,Z,precisions,labels] = sample_HTPLDA_database(nu,F,prior,n);
    fprintf('there are %i speakers\n',max(labels));
    
    [A,b] = HTPLDA.extractSGMEs(R);
    
    rotate = true;
    [Ap,Bp] = SGME.SGME2GME(A,b,rotate);

    close all;
    figure;hold;
    plotGaussian(zeros(zdim,1),eye(zdim),'black, dashed','k--');
    
    %matlab_colours = {'b','r','r'};
    %tikz_colours = {'blue','red','red'};
    %SGME.plotAll(A,b,matlab_colours, tikz_colours, rotate);
    
    
    HTPLDA.plot_database(R,labels,Z);
    axis('square');axis('equal');
    
    calc1 = create_partition_posterior_calculator(SGME.log_expectations,prior,labels);
    calc2 = create_pseudolikelihood_calculator(SGME.log_expectations,prior,labels);
    calc3 = create_BXE_calculator(SGME.log_expectations,[],labels);
    
    scale = exp(-5:0.1:5);
    MCL = zeros(size(scale));
    PsL = zeros(size(scale));
    slowPsL = zeros(size(scale));
    BXE = zeros(size(scale));
    tic;
    for i=1:length(scale)
        MCL(i) = - calc1.logPostPoi(scale(i)*A,scale(i)*b);
    end
    toc

    tic;
    for i=1:length(scale)
        BXE(i) = calc3.BXE(scale(i)*A,scale(i)*b);
    end
    toc
    
    tic;
    for i=1:length(scale)
        slowPsL(i) = - calc2.slow_log_pseudo_likelihood(scale(i)*A,scale(i)*b);
    end
    toc

    tic;
    for i=1:length(scale)
        PsL(i) = - calc2.log_pseudo_likelihood(scale(i)*A,scale(i)*b);
    end
    toc

    
    
    
    figure;
    %subplot(2,1,1);semilogx(scale,MCL);title('MCL')
    %subplot(2,1,2);semilogx(scale,PsL);title('PsL');
    subplot(2,1,1);semilogx(scale,MCL,scale,slowPsL,scale,PsL,'--');legend('MCL','slowPsL','PsL');
    subplot(2,1,2);semilogx(scale,BXE);legend('BXE');
    
    %[precisions;b]
    
    %[plain_GME_log_expectations(Ap,Bp);SGME.log_expectations(A,b)]
    
    
    
end

function test_PsL()

    zdim = 2;
    xdim = 20;      %required: xdim > zdim
    nu = 3;         %required: nu >= 1, integer, DF
    fscal = 3;      %increase fscal to move speakers apart
    
    F = randn(xdim,zdim)*fscal;

    
    HTPLDA = create_HTPLDA_extractor(F,nu);
    SGME = HTPLDA.SGME;
    
    
    n = 1000;
    m = 100;
    %prior = create_PYCRP(0,[],m,n);
    prior = create_PYCRP([],0,m,n);
    [R,Z,precisions,labels] = sample_HTPLDA_database(nu,F,prior,n);
    fprintf('there are %i speakers\n',max(labels));
    
    [A,b] = HTPLDA.extractSGMEs(R);
    
    rotate = true;
    [Ap,Bp] = SGME.SGME2GME(A,b,rotate);

    close all;
    
    if zdim==2 && max(labels)<=12
        figure;hold;
        plotGaussian(zeros(zdim,1),eye(zdim),'black, dashed','k--');

        HTPLDA.plot_database(R,labels,Z);
        axis('square');axis('equal');
    end
    
    tic;calc0 = create_pseudolikelihood_calculator_old(SGME.log_expectations,prior,labels);toc
    tic;calc1 = create_pseudolikelihood_calculator(SGME.log_expectations,prior,labels);toc;
    tic;calc2 = create_BXE_calculator(SGME.log_expectations,[],labels);toc
    
    scale = exp(-5:0.1:5);
    oldPsL = zeros(size(scale));
    PsL = zeros(size(scale));
    BXE = zeros(size(scale));

%     tic;
%     for i=1:length(scale)
%         slowPsL(i) = - calc1.slow_log_pseudo_likelihood(scale(i)*A,scale(i)*b);
%     end
%     toc
    
    tic;
    for i=1:length(scale)
        oldPsL(i) = - calc0.log_pseudo_likelihood(scale(i)*A,scale(i)*b);
    end
    toc

    tic;
    for i=1:length(scale)
        PsL(i) = - calc1.log_pseudo_likelihood(scale(i)*A,scale(i)*b);
    end
    toc
    
    
%     tic;
%     for i=1:length(scale)
%         BXE(i) = calc2.BXE(scale(i)*A,scale(i)*b);
%     end
%     toc

    figure;
    subplot(2,1,1);semilogx(scale,oldPsL,scale,PsL,'--');legend('oldPsL','PsL');
    subplot(2,1,2);semilogx(scale,BXE);title('BXE');
    
    %[precisions;b]
    
    %[plain_GME_log_expectations(Ap,Bp);SGME.log_expectations(A,b)]
    
    
    
end

