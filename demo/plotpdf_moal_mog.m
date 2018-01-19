load('lap_moal8.mat')
lambda=model.lambda;
kappa=model.kappa;
weight=model.weight;

pdfal = 0;
x = -15:0.01:15;
for i = length(lambda)
pdfal = pdfal+weight(i)*pdfald(x,0,lambda(i),kappa(i));
end
load('lap_mog8.mat')
pdfn=0;
sigma = 2.207441687106595;model.Sigma; 
weight = 1;model.weight/sum(model.weight);
for i = length(sigma)
pdfn = pdfn+weight(i)*normpdf(x,0,sqrt(sigma(i)));
end

% rpdf = (-10<= x & x <=10)/20 ;
% rpdf = pdf('Exponential',abs(x),5)/2;
% rpdf = pdf('Chisquare',x+3,3);
rpdf = (0.2*(x<=4&x>=-1)/5)' + 0.6*pdfald(x,0,1,0.5)+ 0.2* normpdf(x',0,1) ;

figure('position',[100,100,500,397])
hold on
% semilogy(x,pdfal,'LineWidth',2,'LineStyle','-')
[q1,q2,q3] = mleald(noi);
plotpdfald(q1,q2,q3,-15,15);
semilogy(x,pdfn,'LineWidth',2,'LineStyle',':')
semilogy(x,rpdf,'LineWidth',2,'LineStyle','--')
