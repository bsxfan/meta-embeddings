function llr_vs_score_plot(tar_scores,nontar_scores,tar_llrs,nontar_llrs)

    [tar_pav,nontar_pav] = opt_loglr(tar_scores,nontar_scores);
    
    
    scores = [tar_scores(:)',nontar_scores(:)'];
    llrs = [tar_llrs(:)',nontar_llrs(:)'];
    pav_scores = [tar_pav(:)',nontar_pav(:)'];
    
    plot(scores,pav_scores,'g',scores,llrs,'r');
    legend('pav','your effort');
    xlabel('score');
    ylabel('llr');
    title('PAV vs your calibration');

end