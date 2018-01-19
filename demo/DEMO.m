m=100;
n=30;
U=randn(m,r);
V=randn(n,r);
X=U*V';

InW = ones(m,n);
for i=1:m*n;
    ind = rand(1);
    if ind<0.3;ald(i) = randald(1,0,4,1.5);end
    if ind>=0.3;ald(i) = randald(1,0,1,1.5);end
end
InX = X+reshape(ald,m,n);
r=4;
param = [];
[label,model,W,U1,V1,llh] = moal(InW,InX,r,param);
X1 = U1*V1';
[sum(sum((X-X1).^2)),sum(sum(abs(X-X1)))]
[subspace(U1,U),subspace(V1,V)]

param.display = 0;
[U2,V2] = CWMmc(InX,InW,r,param);
X2 = U2*V2';
[sum(sum((X-X2).^2)),sum(sum(abs(X-X2)))]
[subspace(U2,U),subspace(V2,V)]
