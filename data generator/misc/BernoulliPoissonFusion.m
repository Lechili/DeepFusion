function Bernoulli = BernoulliPoissonFusion(BernoulliMixture,PoissonPointProcessMixture)

GaussianMixture = [BernoulliMixture,PoissonPointProcessMixture];
[Bernoulli,C] = GaussianFusion(GaussianMixture);

numBernoulliMixture = length(BernoulliMixture);
temp1 = 1;
for i = 1:numBernoulliMixture
    temp1 = temp1*(((1-BernoulliMixture(i).weight)/BernoulliMixture(i).weight)^(1/numBernoulliMixture));
end

numPoissonMixture = length(PoissonPointProcessMixture);
temp2 = 1;
for i = 1:numPoissonMixture
    temp2 = temp2*(PoissonPointProcessMixture(i).weight^(1/numPoissonMixture));
end

Bernoulli.weight = 1/(1/C*temp1/temp2+1);

end

