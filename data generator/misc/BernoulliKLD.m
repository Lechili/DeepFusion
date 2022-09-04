function KLD = BernoulliKLD(Bernoulli0,Bernoulli1)

if Bernoulli0.weight >= 1
    Bernoulli0.weight = 1-eps;
end

if Bernoulli1.weight >= 1
    Bernoulli1.weight = 1-eps;
end

inverseCovariance1 = eye(4)/((Bernoulli1.covariance+Bernoulli1.covariance')/2);
meanDiff = Bernoulli1.mean - Bernoulli0.mean;

KLD = 1/2*(trace(inverseCovariance1*Bernoulli0.covariance) + meanDiff'*inverseCovariance1*meanDiff-4+log(det(Bernoulli1.covariance)/det(Bernoulli0.covariance)));

KLD_r1 = Bernoulli0.weight*log(Bernoulli0.weight/Bernoulli1.weight);
KLD_r2 = (1-Bernoulli0.weight)*log((1-Bernoulli0.weight)/(1-Bernoulli1.weight));

KLD = KLD + KLD_r1 + KLD_r2;


end

