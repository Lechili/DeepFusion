function [singleGaussian,C] = GaussianFusion(GaussianMixture)

numGaussian = length(GaussianMixture);
weights = [GaussianMixture.weight];
sumWeights = sum(weights);
weights = weights/sumWeights;

singleGaussian.weight = sumWeights;

inverseCovariance = zeros(4,4,numGaussian);
for i = 1:numGaussian
    inverseCovariance(:,:,i) = eye(4)/((GaussianMixture(i).covariance+GaussianMixture(i).covariance')/2);
end

temp = zeros(4,4);
for i = 1:numGaussian
    temp = temp + weights(i)*eye(4)*inverseCovariance(:,:,i);
end
singleGaussian.covariance = eye(4)/temp;

temp = zeros(4,1);
for i = 1:numGaussian
    temp = temp + weights(i)*eye(4)*inverseCovariance(:,:,i)*GaussianMixture(i).mean;
end
singleGaussian.mean = singleGaussian.covariance*temp;

temp = 0;
for i = 1:numGaussian
    temp = temp + weights(i)*GaussianMixture(i).mean'*inverseCovariance(:,:,i)*GaussianMixture(i).mean;
end

C = sqrt(det(2*pi*singleGaussian.covariance))*exp(1/2*(singleGaussian.mean'/singleGaussian.covariance*singleGaussian.mean-temp));
for i = 1:numGaussian
    C = C/(det(2*pi*GaussianMixture(i).covariance)^(weights(i)/2));
end


end

