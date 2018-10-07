function demo_adaptSPLDA()

    % Syntesize old SPLDA model 
    big = false;
    [oldF,oldW] = simulateSPLDA(big);
    
    % Perturb to get ne model and generate data
    if big
        num_new_Fcols = 10;
        W_adj_rank = 10;
    else
        num_new_Fcols = 2;
        W_adj_rank = 2;
    end        
    nspeakers = 1000;
    recordings_per_speaker = 5;
    [newF,newW,newData,newLabels] = perturb_and_simulate_SPLDA(oldF,oldW,...
                         num_new_Fcols,W_adj_rank,...
                         nspeakers,recordings_per_speaker);

    oracle_obj = - splda_llh_full(newLabels,newF,newW,newData),
    old_obj = - splda_llh_full(newLabels,oldF,oldW,newData),

%     scale = exp(-5:0.1:20);
%     n = length(scale);
%     y = zeros(1,n);
%     for i=1:n
%         y(i) = - splda_llh_full(newLabels,scale(i)*newF,newW,newData);
%     end
%     semilogx(scale,y);
    
    
    %return;
    
    
                     
    maxiters = 1000;
    timeout = 10*60;
    [F,W] = train_ML_adapted_SPLDA(oldF,oldW,newData,newLabels,...
                                              num_new_Fcols,W_adj_rank, ...
                                              maxiters,timeout);
                     
    oracle_obj = - splda_llh_full(newLabels,newF,newW,newData),
    final_obj = - splda_llh_full(newLabels,F,W,newData),
                     
    
end

