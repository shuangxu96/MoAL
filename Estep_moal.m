function [R, llh] = Estep_moal(Error, model)
% Compute the conditional probability
alpha = model.alpha;
lambda = model.lambda;
kappa = model.kappa;
w = model.weight;

n = size(Error,2);
k = size(alpha,2);
logRho = zeros(n,k);

for i = 1:k
    logRho(:,i) = logpdfald(Error,alpha(i),lambda(i),kappa(i));
end
logRho = bsxfun(@plus,logRho,log(w));
T = logsumexp(logRho,2);
llh = sum(T)/n; % loglikelihood
logR = bsxfun(@minus,logRho,T);
R = exp(logR);
