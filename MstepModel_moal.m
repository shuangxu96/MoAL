function [model,rho] = MstepModel_moal(Error,R,model)
kappa = model.kappa;
k = size(R,2);
alpha = zeros(1,k);%fix mu to zero
% weight
nk = sum(R,1);
w = nk/size(R,1);
% lambda
inx = Error<0;
rho = inx*(1-kappa) + ~inx*kappa;
lambda = nk./  sum(R.*rho.*abs(Error));
indINF = find(~isfinite(lambda));
if ~isempty(indINF)
    lambda(indINF) = nk(indINF)/1e-6;
end
% kappa
eta = sum(R.*Error).*lambda;
sDelta = sqrt(4*nk.^2+eta.^2);
kappa = (2*nk + eta - sDelta) ./ (2*eta);

model.alpha = alpha;
model.lambda = lambda;
model.kappa = kappa;
model.weight = w;
