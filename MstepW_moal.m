function [W, TempU, TempV] = MstepW_moal(InW,InX,TempU,TempV,R,NumIter,rho,model)
IND = find(InW(:)~=0);
r = size(TempU,2);
W = zeros(size(InX));
lambda = model.lambda;

W(IND) = sum(R.*rho.*lambda,2);
% W = (W-min(min(W)))/(max(max(W))-min(min(W)));
pa = [];
pa.IniU = TempU;
pa.IniV = TempV;
pa.maxiter = NumIter;
pa.display = 0;
[TempU,TempV] = CWM_wei(InX,W,r,pa);
