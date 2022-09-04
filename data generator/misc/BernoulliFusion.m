function Bernoulli = BernoulliFusion(BernoulliMixture)

[Bernoulli,C] = GaussianFusion(BernoulliMixture);

numBernoulliMixture = length(BernoulliMixture);
temp = 1/C;
for i = 1:numBernoulliMixture
    temp = temp*(((1-BernoulliMixture(i).weight)/BernoulliMixture(i).weight)^(1/numBernoulliMixture));
end

Bernoulli.weight = 1/(temp+1);

end

