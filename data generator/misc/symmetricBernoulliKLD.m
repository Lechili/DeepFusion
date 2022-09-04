function KLD = symmetricBernoulliKLD(Bernoulli0,Bernoulli1)

KLD = 1/2*(BernoulliKLD(Bernoulli0,Bernoulli1)+BernoulliKLD(Bernoulli1,Bernoulli0));

end

