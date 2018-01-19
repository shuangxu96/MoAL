function [OutU,OutV,model,label,W,llh] = moal(InW,InX,r,param)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Shuang Xu, Chunxia Zhang. Adaptive Quantile Low-rank Matrix  %
% Factorization, CVPR, 2018                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%USAGE:[label,model,W,OutU,OutV,llh] = moal(InW,InX,r,param)
%Input:
%   InX: d x n input data matrix
%   InW: d x n indicator matrix
%   r:   the rank
%   param.maxiter: maximal iteration number
%   param.OriX: ground truth matrix
%   param.InU,InV: Initialized factorized matrices
%   param.k: the number of GMM
%   param.display: display the iterative process
%   param.tol: the thresholding for stop
%Output:
%   label:the labels of the noises
%   model:model.alpha, the means of the different Gaussians
%         model.sigma,the variance of the different Gaussians
%         model.weight,the mixing coefficients
%   W: d x n weighted matrix
%   OutU: the fianl factorized matrix U
%   OutV: the fianl factorized matrix V
%   llh:  the log likelihood
%
% Author: Shuang Xu(shuangxu@stu.xjtu.edu.cn)
% Date: 01-31-2018


%% initialization
[m, n] = size(InX);
IND = find(InW(:) ~= 0);
% process param
if (~isfield(param,'maxiter'));maxiter = 100;else;maxiter = param.maxiter;end
if (~isfield(param,'OriX'));OriX = InX;else;OriX = param.OriX;clear param.OriX;end
if (~isfield(param,'k'));k = 5;else;k = param.k;end
if (~isfield(param,'display'));display = 1;else;display = param.display;end
if (~isfield(param,'NumIter'));NumIter = 100;else;NumIter = param.NumIter;end
if (~isfield(param,'tol'));tol = 1.0e-50;else;tol = param.tol;end
if (~isfield(param,'tuningK'));tuningK = true;else;tuningK = param.tuningK;end
if (~isfield(param,'InU'))
    s = median(abs(InX(IND)));
    s = sqrt(s/r);
    if min(InX(IND)) >= 0
        InU = rand(m,r)*s;
    else
        InU = rand(m,r)*s*2-s;
    end
else
    InU = param.InU;
end

if (~isfield(param,'InV'))
    if min(InX(IND)) >= 0
        InV = rand(n,r)*s;
    else
        InV = rand(n,r)*s*2-s;
    end
else
    InV = param.InV;
end

tempX=InX(IND);
R = initialization_moal(tempX',k); % randomize the class label / gamma
[~,label(1,:)] = max(R,[],2);
R = R(:,unique(label)); % delete blank class
%%Initialize MoAL parameters
if (~isfield(param,'model'))
    model.alpha = zeros(1,k);
    model.lambda = rand(1,k);
    model.kappa = rand(1,k);
    nk = sum(R,1);
    model.weight = nk/size(R,1);
else
    model = param.model;
end

TempU = InU;
TempV = InV;
TempX = TempU*TempV';
Error = InX(:)-TempX(:);
Error = Error(IND);

t = 1;
[R, llh(t)] = Estep_moal(Error',model);
converged = false;
while ~converged && t < maxiter
    t = t+1;
    %%%%%%%%%%%%%%%%1. M Step 1 %%%%%%%%%%%%%%%%%%%
    [model,rho] = MstepModel_moal(Error,R,model);
    %%%%%%%%%%%%%%%%2. E Step %%%%%%%%%%%%%%%%%%%
    [R, llh(t)] = Estep_moal(Error',model);
    L1 = llh(t);
    %%%%%%%%%%%%%%%%3. M Step 2 %%%%%%%%%%%%%%%%%%%
    [W, TempU, TempV] = MstepW_moal(InW,InX,TempU,TempV,R,NumIter,rho,model);
    TempX = TempU*TempV';
    Error = InX(:)-TempX(:);
    Error = Error(IND);
    %%%%%%%%%%%%%%%%4. E Step %%%%%%%%%%%%%%%%%%%
    [R, llh(t)] = Estep_moal(Error',model);
    L2 = llh(t);
    %%%%%%%%%%%%%%%%5. display %%%%%%%%%%%%%%%%%%%%
    display_moal(L1,L2,k,model,OriX,TempU,TempV,InW,Error,param,display);
    %%%%%%%%%%%%%%%%6. tune k %%%%%%%%%%%%%%%%%%%%
    old_k = k;
    if (tuningK)
        [~,label(:)] = max(R,[],2);
        u = unique(label);   % non-empty components
        if size(R,2) ~= size(u,2)
            R = R(:,u);   % remove empty components
            model.alpha = model.alpha(:,u);
            model.lambda = model.lambda(:,u);
            model.kappa = model.kappa(:,u);
            model.weight = model.weight(:,u);
        end
        k = length(u);
    end
    %%%%%%%%%%%%%%%%7. Converge? %%%%%%%%%%%%%%%%%%%%
    if k~=old_k
        converged = false;
    elseif sum(model.weight)~=1
        converged = false;
    else
        %converged = norm(OldU - TempU)<tol;
        converged = abs(L2-llh(t-1))<tol;
    end
end
OutU = TempU;
OutV = TempV;
display_moal(L1,L2,k,model,OriX,TempU,TempV,InW,Error,param,display);

if converged
    fprintf('Converged in %d steps.\n',t-1);
else
    fprintf('Not converged in %d steps.\n',maxiter);
end
end

%% sub-functions
function display_moal(L1,L2,k,model,OriX,TempU,TempV,InW,Error,param,display)
if display
    disp(['The likelihood in this step is ',num2str(L1),' and ',num2str(L2),';']);
    disp(['There are ',num2str(k),' ALD noises mixed in data']);
    disp(['with lambda ',num2str(model.lambda)]);
    disp(['with kappa ',num2str(model.kappa)]);
    disp(['with weights ',num2str(model.weight),'.']);
    if (~isfield(param,'orimissing'))
        disp(['Relative reconstruction error ', num2str(sum(sum(((OriX - TempU*TempV')).^2)))]);
    else
        disp(['Relative reconstruction error ', num2str(sum(sum((InW.*(OriX - TempU*TempV')).^2)))]);
    end
    disp(['L2 RMSE is ', num2str(sqrt(sum(Error.^2)))]);
    disp(['L1 RMSE is ', num2str(sum(abs(Error)))]);
end
end % end function display_moal