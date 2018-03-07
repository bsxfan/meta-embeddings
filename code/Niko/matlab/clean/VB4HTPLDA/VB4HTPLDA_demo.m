function VB4HTPLDA_demo
% Demo and test code for VB training and SGME scoring of HT-PLDA model.
%
% Training and evaluation data are (independently) sampled from a model with 
% randomly generated data. A VB algorithm is used to estimate the parameters
% of this model from the training data. The accuracy of the trained (VB) 
% model is compared (on both train and evaluation data) against the 
% (oracle) model that generated the data.
%
% The accuracy is given in terms of the calibration-sensitive binary cross
% entropy (BXE) and (if available) also equal-error-rate EER.
% 
% If the BOSARIS Toolkit (https://sites.google.com/site/bosaristoolkit/) is
% available, BXE is shown not only for the 'raw' scores as given by this 
% model, but also for PAV-recalibrated scores, to give 'minBXE'. The latter
% is what BXE could have been if calibration had been ideal.



    % Assemble model to generate data
    zdim = 2;       %speaker identity variable size 
    rdim = 20;      %i-vector size. required: rdim > zdim
    nu = 3;         %required: nu >= 1, integer, degrees of freedom for heavy-tailed channel noise
    fscal = 3;      %increase fscal to move speakers apart
    F = randn(rdim,zdim)*fscal;
    W = randn(rdim,rdim+1);W = W*W.'; W = (rdim/trace(W))*W;
    model1 = create_HTPLDA_SGME_backend(nu,F,W);  %oracle model
    
    
    %Generate synthetic labels
    nspeakers = 1000;
    recordings_per_speaker = 10;
    N = nspeakers*recordings_per_speaker;
    ilabels = repmat(1:nspeakers,recordings_per_speaker,1);
    ilabels = ilabels(:).';  % integer speaker labels
    hlabels = sparse(ilabels,1:N,true,nspeakers,N); %speaker label matrix with one-hot columns
    
    %and some training data
    Z = randn(zdim,nspeakers);
    Train = F*Z*hlabels + sample_HTnoise(nu,rdim,N,W);
    
    
    
    
    %train
    fprintf('*** Training on %i i-vectors of %i speakers ***\n',N,nspeakers);
    niters = 10;
    [model2,obj] = HTPLDA_SGME_train_VB(Train,hlabels,nu,zdim,niters);
    close all;
    plot(obj);title('VB lower bound');
    
    
    %Generate independent evaluation data with new speakers
    nspeakers = 300;
    
    %Generate target speakers
    ntar = nspeakers;
    Ztar = randn(zdim,ntar);

    %and some single enrollment data for them
    Enroll1 = F*Ztar + sample_HTnoise(nu,rdim,ntar,W);   %1 enrollment / speaker
    
    %and some double enrollments
    ne = 2;
    Flags = repmat(1:ntar,ne,1);  
    Flags = sparse(Flags(:),1:ne*ntar,true,ntar,ne*ntar);
    Enroll2 = F*Ztar*Flags + sample_HTnoise(nu,rdim,ne*ntar,W);   %2 enrollments / speaker
    
    %and some test data
    recordings_per_speaker = 10;
    N = nspeakers*recordings_per_speaker;
    ilabels = repmat(1:nspeakers,recordings_per_speaker,1);
    ilabels = ilabels(:).';  % integer speaker labels
    hlabels = sparse(ilabels,1:N,true,nspeakers,N); %speaker label matrix with one-hot columns
    Test = F*Ztar*hlabels + sample_HTnoise(nu,rdim,N,W);

    fprintf('\n\n*** Evaluation on %i target speakers with single/double enrollments and %i test recordings ***\n',nspeakers,N);

    useBOSARIS = exist('opt_loglr','file');
    
    if useBOSARIS
        fprintf('  minBXE in brackets\n')    
    
        BXE = zeros(2,2);
        minBXE = zeros(2,2);
        EER = zeros(2,2);
        Scores = model1.score_trials(Enroll1,[],Test);
        [BXE(1,1),minBXE(1,1),EER(1,1)] = calcBXE(Scores,hlabels);
        Scores = model1.score_trials(Enroll2,Flags,Test);
        [BXE(1,2),minBXE(1,2),EER(1,2)] = calcBXE(Scores,hlabels);
        Scores = model2.score_trials(Enroll1,[],Test);
        [BXE(2,1),minBXE(2,1),EER(2,1)] = calcBXE(Scores,hlabels);
        Scores = model2.score_trials(Enroll2,Flags,Test);
        [BXE(2,2),minBXE(2,2),EER(2,2)] = calcBXE(Scores,hlabels);

        fprintf('oracle: single enroll BXE = %g (%g), double enroll BXE = %g (%g)\n',BXE(1,1),minBXE(1,1),BXE(1,2),minBXE(1,2));
        fprintf('VB    : single enroll BXE = %g (%g), double enroll BXE = %g (%g)\n',BXE(2,1),minBXE(2,1),BXE(2,2),minBXE(2,2));

        fprintf('oracle: single enroll EER = %g, double enroll EER = %g\n',EER(1,1),EER(1,2));
        fprintf('VB    : single enroll EER = %g, double enroll EER = %g\n',EER(2,1),EER(2,2));
    
    else  % no BOSARIS

        BXE = zeros(2,2);
        Scores = model1.score_trials(Enroll1,[],Test);
        BXE(1,1) = calcBXE(Scores,hlabels);
        Scores = model1.score_trials(Enroll2,Flags,Test);
        BXE(1,2) = calcBXE(Scores,hlabels);
        Scores = model2.score_trials(Enroll1,[],Test);
        BXE(2,1) = calcBXE(Scores,hlabels);
        Scores = model2.score_trials(Enroll2,Flags,Test);
        BXE(2,2) = calcBXE(Scores,hlabels);

        fprintf('oracle: single enroll BXE = %g, double enroll BXE = %g\n',BXE(1,1),BXE(1,2));
        fprintf('VB    : single enroll BXE = %g, double enroll BXE = %g\n',BXE(2,1),BXE(2,2));
    end
        
end


function [bxe,min_bxe,EER] = calcBXE(Scores,labels)
% Binary cross-entropy, with operating point at target prior at true
% proportion, normalized so that llr = 0 gives bxe = 1.
    tar = Scores(labels);
    non = Scores(~labels);
    ofs = log(length(tar)) - log(length(non));
    bxe = mean([softplus(-tar - ofs).',softplus(non + ofs).']) / log(2);

    if nargout>=2
        [tar,non] = opt_loglr(tar.',non.','raw');
        tar = tar';
        non = non.';
        min_bxe = mean([softplus(-tar - ofs).',softplus(non + ofs).']) / log(2);
    end
    
    if nargout >=3
        EER = eer(tar,non);
    end
end

function y = softplus(x)
% y = log(1+exp(x));
    y = x;
    f = find(x<30);
    y(f) = log1p(exp(x(f)));
end

function X = sample_HTnoise(nu,dim,n,W)
% Sample n heavy-tailed dim-dimensional variables. (Only for integer nu.)
%
% Inputs:
%   nu: integer nu >=1, degrees of freedom of resulting t-distribution
%   n: number of samples
%   W: precision matrix for T-distribution
%
% Output:
%   X: dim-by-n samples

    cholW = chol(W);    
    precisions = mean(randn(nu,n).^2,1);  
    std = 1./sqrt(precisions);
    X = cholW\bsxfun(@times,std,randn(dim,n));
end


