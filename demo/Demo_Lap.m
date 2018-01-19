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

num = 10;
load('IND.mat')
step=1;
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
    U = rand(1,m*n-p1)-0.5+1e-10;
    lap = -5*sign(U).*log(1-2*abs(U));
    X_Noi(Ind(p1+1:end)) = X_Noi(Ind(p1+1:end)) + lap;
    random('t',1,1,m*n-p1);  %Add Gaussian noise
    
    aa = median(abs(X_Noi(Ind(p1+1:end))));
    aa = sqrt(aa/r);
    for i = 1:num
        U0 = rand(m,r)*aa*2-aa;
        V0 = rand(n,r)*aa*2-aa;
        
        %%%%%%%%%%%%%%%%%%MoAL method %%%%%%%%%%%%%%%%%%%%%%%%
        param.maxiter = 100;
        param.OriX = X_Ori;
        param.InU = U0;
        param.InV = V0;
        param.k = 4;
        param.display = 0;
        param.NumIter = 50;
        param.tol = 1.0e-3;
        
        [A,B,model,label,W,llh] = moal(W,X_Noi,r,param);
%                 [label, model,TW, A,B,llh] =  MLGMDN(W,X_Noi,r,param);
                       M{step} = model;
        LL(step) = llh(end);
        step = step+1;
%                 [M_est, A, B, L1_error] = RobustApproximation_M_UV_TraceNormReg(X_Noi,W,r);
%                 B = B';
%                 [A,B] = CWMmc(X_Noi,W,r,param);
%                 [U, V, it, lambda, invZ, EZ] = VBMFL1_II(X_Noi, W, r, U0', V0', 100);
%                 A = U'; B = V';
%         [U, V, err, iter, err_log, lambda_log] = damped_wiberg_new(X_Noi, W, r, V0, 1e-3, 100, 0);
%         A = U; B=V;
        E1G(kk,i) = sum(sum(abs((X_Ori-A*B'))));
        E2G(kk,i) = sum(sum(((X_Ori - A*B')).^2));
        E3G(kk,i) = subspace(RU,A);
        E4G(kk,i) = subspace(RV',B);
%         SG(kk,i) = llh(end);
        
    end
end






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