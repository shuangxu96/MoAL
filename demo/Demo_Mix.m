%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deyu Meng, Fernando De la Torre. Robust matrix factorization %
% with unknown noise, ICCV, 2013                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo on data with Gaussian noise experiments                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
clc;

m = 40;
n = 20;                     %Data size
r = 8;                      %Rank
step=1;
num = 10;
load('IND.mat')
for kk = 1:30
    kk
    randn('seed',kk),RU = randn(m,r);
    randn('seed',kk),RV = randn(r,n);
    X_Ori = RU * RV;        %Original data
    Ind = IND(kk,:);
    p1 = floor(m*n*0.2);
    W = ones(m,n);
    W(Ind(1:p1)) = 0;       %Indicator matrix
    X_Noi = X_Ori;
    X_Noi = W.*X_Noi;       %Add missing components
    noi=[];
    for q=1:(m*n-p1)
        u = rand(1);
        if u<0.2; noi(q) = 5*(rand(1)-0.2); % [-1,4]
        elseif u>=0.2&u<0.4;noi(q) = randn(1);
        else u>=0.4;noi(q) = randald(1,0,1,0.5);end
    end
    X_Noi(Ind(p1+1:end)) = X_Noi(Ind(p1+1:end)) + noi;
    
    
    aa = median(abs(X_Noi(Ind(p1+1:end))));
    aa = sqrt(aa/r);
    
    for i = 1:num
        U0 = rand(m,r)*aa*2-aa;
        V0 = rand(n,r)*aa*2-aa;
        
        %%%%%%%%%%%%%%%%%%MoAL method %%%%%%%%%%%%%%%%%%%%%%%%
        param.maxiter = 100;
        param.OriX = X_Ori;
        
        param.k = 4;
        param.display = 0;
        param.NumIter = 50;
        param.tol = 1.0e-3;
        param.InU = U0;
        param.InV = V0;
        [A,B,model,label,W,llh] = moal(W,X_Noi,r,param); 
%                 [label, model,TW, A,B,llh] =  MLGMDN(W,X_Noi,r,param);
                       M{step} = model;
        LL(step) = llh(end);
        step = step+1;
%                 [M_est, A, B, L1_error] = RobustApproximation_M_UV_TraceNormReg(X_Noi,W,r);
%                 B = B';
%                                 [A,B] = CWMmc(X_Noi,W,r,param);
        %                         [U, V, it, lambda, invZ, EZ] = VBMFL1_II(X_Noi, W, r, U0', V0', 100);
        %                         A = U'; B = V';
%                         [U, V, err, iter, err_log, lambda_log] = damped_wiberg_new(X_Noi, W, r, V0, 1e-3, 100, 0);
%                         A = U; B=V;
        E1G(kk,i) = sum(sum(abs((X_Ori-A*B'))));
        E2G(kk,i) = sum(sum(((X_Ori - A*B')).^2));
        E3G(kk,i) = subspace(RU,A);
        E4G(kk,i) = subspace(RV',B);
        %         SG(kk,i) = llh(end);

    end
end

% lambda=model.lambda;
% kappa=model.kappa;
% weight=model.weight;
% pdf = 0;
% x = -10:0.01:10;
% for i = length(lambda)
% pdf = pdf+weight(i)*pdfald(x,0,lambda(i),kappa(i));
% end
% pdfn=0;sigma = 1.482;model.Sigma; 
% weight = 1;model.weight/sum(model.weight);
% for i = length(sigma)
% pdfn = pdfn+weight(i)*normpdf(x,0,sqrt(sigma(i)));
% end
% rpdf = (0.25*(x<=4&x>=-1)/5)' + 0.25*pdfald(x,0,1,0.5)+ 0.5* normpdf(x',0,1) ;
% figure,hold on
% plot(x,pdf,'LineWidth',2,'LineStyle','-')
% plot(x,rpdf,'LineWidth',2,'LineStyle','--')
% plot(x,pdfn,'LineWidth',2,'LineStyle',':')

for kk = 1:30
    %%%%%%%%%%%%%%%%%%GMM method %%%%%%%%%%%%%%%%%%%%%%%%
    %     [a ii] = max(SG(kk,1:num));
    
    %     EE2G(kk) = E2G(kk,ii);
    %     EE1G(kk) = E1G(kk,ii);
    %     EE3G(kk) = E3G(kk,ii);
    %     EE4G(kk) = E4G(kk,ii);
    %         EE2G(kk) = min(E2G(kk,1:3));
    %     EE1G(kk) = min(E1G(kk,1:3));
    %     EE3G(kk) = min(E3G(kk,1:3));
    %     EE4G(kk) = min(E4G(kk,1:3));
    EE2G(kk) = min(E2G(kk,:));
    EE1G(kk) = min(E1G(kk,:));
    EE3G(kk) = min(E3G(kk,:));
    EE4G(kk) = min(E4G(kk,:));
end

format long
[mean(EE1G)
mean(EE2G)
mean(EE3G)
mean(EE4G)]

[~,inx] = max(LL);
model = M{inx};