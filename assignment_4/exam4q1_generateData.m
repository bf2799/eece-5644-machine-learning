function [x,y] = exam4q1_generateData(N)
close all,
x = gamrnd(3,2,1,N);
z = exp((x.^2).*exp(-x/2));
v = lognrnd(0,0.1,1,N);
y = v.*z;
figure(1), plot(x,y,'.'),
xlabel('x'); ylabel('y');
