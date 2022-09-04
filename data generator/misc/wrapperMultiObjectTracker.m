function [trajectoryEstimate,marginalPMB] = wrapperMultiObjectTracker(measurements, ...
                                                                      parameters, ...
                                                                      T_pruning, ...
                                                                      T_pruningPois, ...
                                                                      Nhyp_max, ...
                                                                      gating_threshold, ...
                                                                      existence_threshold, ...
                                                                      T_alive , ...
                                                                      existence_estimation_threshold1, ...
                                                                      H)



Nsteps = length(measurements);
Lscan = Nsteps;

R = parameters.measurementNoiseCov;
p_d = parameters.detectionProbability;
intensity_clutter = parameters.PoissonClutterIntensity;
F = parameters.transitionMatrix;
Q = parameters.motionProcessNoiseCov;
p_s = parameters.survivalProbability;

weights_b = parameters.PoissonBirthRate;
means_b = parameters.PoissonBirthGaussianMean;
covs_b = parameters.PoissonBirthGaussianCov;

filter_pred = cell(0,1);
filter_pred.Pois{1}.weightPois = parameters.PoissonBirthRate;
filter_pred.Pois{1}.meanPois = parameters.PoissonBirthGaussianMean;
filter_pred.Pois{1}.covPois = parameters.PoissonBirthGaussianCov;
filter_pred.Pois{1}.t_bPois = 1;
filter_pred.Pois{1}.length_Pois = 1;

filter_pred.tracks = cell(0,1);
filter_pred.globHyp = [];
filter_pred.globHypWeight = [];
N_hypotheses_t = zeros(1,Nsteps);

for k = 1:Nsteps
    z = measurements{k};

    filter_upd = TPMBM_all_update(filter_pred,z,H,R,p_d,k,gating_threshold,intensity_clutter,Nhyp_max,Lscan,T_alive);
    filter_upd_pmb = TPMB_all_projection(filter_upd,T_alive,4,k,Lscan);
    filter_upd = filter_upd_pmb;

    [X_estimate,X_cov_estimate,X_length_estimate,t_b_estimate,length_estimate,existence_estimate,alive_prob] = TPMBM_all_estimate1(filter_upd,existence_estimation_threshold1,4,k);

    filter_upd_pruned = TPMBM_all_pruning(filter_upd,T_pruning,T_pruningPois,Nhyp_max,existence_threshold,T_alive);
    filter_upd = filter_upd_pruned;
    N_hypotheses_t(k) = length(filter_upd.globHypWeight);

    filter_pred = TPMBM_all_prediction(filter_upd,F,Q,p_s,weights_b,means_b,covs_b,Lscan,k,T_alive);
end

marginalPMB = PMB_estimate(filter_upd,T_alive);

trajectoryEstimate.existence_estimate = existence_estimate;
trajectoryEstimate.startTime = t_b_estimate;
trajectoryEstimate.length = length_estimate;
trajectoryEstimate.state = X_estimate;
trajectoryEstimate.covariance = X_cov_estimate;
trajectoryEstimate.lengthProbability = X_length_estimate;
trajectoryEstimate.alive_prob = alive_prob;
end

