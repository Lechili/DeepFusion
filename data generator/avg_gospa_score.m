function [add_info,MT3_GOSPA,SOTA_GOSPA] = avg_gospa_score(n_pred,EVAL,ValData,p,c,alpha,ndim)

%This function computes the average GOSPA and NLL scores for MT3v2 and the
%Bayesian mothod, together with their decompositions.

MT3_scores = 0;
SOTA_scores = 0;

MT3_loc_errors = 0;
MT3_missed_errors = 0;
MT3_false_errors = 0;

SOTA_loc_errors = 0;
SOTA_missed_errors = 0;
SOTA_false_errors = 0;



% Store the number of empty labels.
empty_count = 0;

% Store the number of empty SOTA labels
n_empty_SOTA = 0;

% Store the number of empty predicted labels
n_empty_MT3 = 0;

% Store the index of empty labels.
empty_index = [];

% Store the index of empty SOTA labels.
empty_SOTA_index =[];

% Store the index of empty predicted labels.
empty_MT3_index = [];

for i =  1: n_pred
    if isempty(ValData.GT{i,1}{1,1})
        labels = zeros(ndim,0);
        empty_count = empty_count + 1;
        empty_index = [empty_index i];
    else
        labels = ValData.GT{i,1}{1,1}(:,1:ndim);

    BernoulliEstimates = ValData.SOTA{i,1}{1,1};
    Y = [BernoulliEstimates.mean];
    if isempty(Y)
        Y = zeros(ndim,0);
        n_empty_SOTA = n_empty_SOTA +1 ;
        empty_SOTA_index =[empty_SOTA_index i];
        [d_gospa_SOTA,~, SOTA_decomposed_cost] = GOSPA(Y, labels', p, c, alpha);
    else
        Y = Y(1:ndim,:);
        [d_gospa_SOTA,~, SOTA_decomposed_cost] = GOSPA(Y, labels', p, c, alpha);
    end

    if isempty(EVAL{i,1})
        MT3 = zeros(ndim,0);
        n_empty_MT3 = n_empty_MT3 +1 ;
        [d_gospa_MT3,~, MT3_decomposed_cost] = GOSPA(MT3, labels', p, c, alpha);
        empty_MT3_index = [empty_MT3_index i];
    else
        MT3 = EVAL{i,1}(:,1:ndim);
        [d_gospa_MT3,~, MT3_decomposed_cost] = GOSPA(MT3', labels', p, c, alpha);
    end


    MT3_scores = MT3_scores +d_gospa_MT3;
    SOTA_scores = SOTA_scores +d_gospa_SOTA;

    MT3_loc_errors = MT3_loc_errors + MT3_decomposed_cost.localisation;
    MT3_missed_errors = MT3_missed_errors +MT3_decomposed_cost.missed;
    MT3_false_errors = MT3_false_errors + MT3_decomposed_cost.false;
    
    SOTA_loc_errors = SOTA_loc_errors + SOTA_decomposed_cost.localisation;
    SOTA_missed_errors = SOTA_missed_errors + SOTA_decomposed_cost.missed;
    SOTA_false_errors = SOTA_false_errors + SOTA_decomposed_cost.false;
    end
end

%actual = n_pred - empty_count;
%add_info.actual = actual;
add_info.empty_index = empty_index;
add_info.empty_SOTA_index = empty_SOTA_index;
add_info.empty_MT3_index = empty_MT3_index;
add_info.n_empty_SOTA = n_empty_SOTA;
add_info.n_empty_MT3 = n_empty_MT3;

MT3_GOSPA.MT3_scores = MT3_scores / n_pred;
SOTA_GOSPA.SOTA_scores = SOTA_scores /n_pred;

MT3_GOSPA.MT3_loc_errors = MT3_loc_errors / n_pred;
MT3_GOSPA.MT3_missed_errors = MT3_missed_errors / n_pred;
MT3_GOSPA.MT3_false_errors = MT3_false_errors /n_pred;

SOTA_GOSPA.SOTA_loc_errors = SOTA_loc_errors /n_pred;
SOTA_GOSPA.SOTA_missed_errors = SOTA_missed_errors /n_pred;
SOTA_GOSPA.SOTA_false_errors = SOTA_false_errors /n_pred;

end