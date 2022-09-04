function marginalPMB = PMB_estimate(filter_upd,T_alive)

numPoissonGaussian = length(filter_upd.Pois);

marginalPMB.Poisson = repmat(struct('weight',[],'mean',[],'covariance',[]),1,numPoissonGaussian);

for i = 1:numPoissonGaussian
    marginalPMB.Poisson(i).weight = filter_upd.Pois{i}.weightPois;
    marginalPMB.Poisson(i).mean = filter_upd.Pois{i}.meanPois(end-3:end);
    marginalPMB.Poisson(i).covariance = filter_upd.Pois{i}.covPois(end-3:end,end-3:end);
end

numBernoulli = length(filter_upd.tracks);

marginalPMB.multiBernoulli = repmat(struct('weight',[],'mean',[],'covariance',[]),1,numBernoulli);

for i = 1:numBernoulli
    marginalPMB.multiBernoulli(i).weight = filter_upd.tracks{i}.eB*filter_upd.tracks{i}.prob_length{1}(1);
    marginalPMB.multiBernoulli(i).mean = filter_upd.tracks{i}.meanB{1}(end-3:end);
    marginalPMB.multiBernoulli(i).covariance = filter_upd.tracks{i}.covB{1}(end-3:end,end-3:end);
end

if length(marginalPMB.multiBernoulli) > 1
    idx_keep = [marginalPMB.multiBernoulli.weight] >= T_alive;
    marginalPMB.multiBernoulli = marginalPMB.multiBernoulli(idx_keep);
end

end