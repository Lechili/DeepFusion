function PoissonPointProcess = PoissonFusion(PoissonPointProcessMixture)

[PoissonPointProcess,C] = GaussianFusion(PoissonPointProcessMixture);

numPoissonMixture = length(PoissonPointProcessMixture);
PoissonPointProcess.weight = C;
for i = 1:numPoissonMixture
    PoissonPointProcess.weight = PoissonPointProcess.weight*(PoissonPointProcessMixture(i).weight^(1/numPoissonMixture));
end

end

