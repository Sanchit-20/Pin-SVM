% Author: Rahul Choudhary & Sanchit Jalan

%------Generating artificial 2d two class dataset-------


n = 100;	%Number of samples in each class

% mu1 = [0.5 -3];
% sigma1 = [0.2 0; 0 3];

% obj1 = mvnrnd(mu1, sigma1, n);

% mu2 = [-0.5 3];
% sigma2 = [0.2 0; 0 3];

% obj2 = mvnrnd(mu2, sigma2, n);

% save('data_without_noise.mat', 'obj1', 'obj2');


load('data_without_noise.mat');
load('noise.mat');

whos('-file', 'data_without_noise.mat');
whos('-file', 'noise.mat');

plot(obj1(:, 1), obj1(:, 2), '+');
hold on;
plot(obj2(:, 1), obj2(:, 2), 'o');

X = [obj1; obj2; noise];
Y = [ones(n, 1); -ones(n, 1); Y_noise];

noise_size = size(Y_noise);
noise_level=noise_size(1)/n;
s=num2str(noise_level);

n = 100 + (noise_size(1)/2);


woptimal=[];
boptimal=[];

%--------Hinge loss SVM classifier---------

H = diag([0 1 1 zeros(1, 2*n)]);
gamma = 1;
f = [zeros(3, 1); gamma*ones(2*n, 1)];
Aeq = []
Beq = []
A = [-Y -X.*(Y*[1 1]) -eye(2*n)];
B = -ones(2*n, 1);
lb = [-inf*ones(3, 1); zeros(2*n, 1)];
% ub = [inf*ones(3 + 2*n, 1)];
ub = [inf*ones(2*n+3,1)];

[W, fval] = quadprog(H, f, A, B, Aeq, Beq, lb, ub);
b = W(1, :);
w = W(2:3, :);

woptimal =[woptimal;w];
boptimal=[boptimal,b];

disp(b);
disp(w);

disp('-w(1)/w(2) for C-SVM:');
disp(-(w(1)/w(2)));

disp('-b/w(2) for C-SVM:');
disp(-(b/w(2)));

%--------Plotting the optimal Hinge loss hyperplane--------

x = -2:0.1:2;
y1 = -(b + w(1)*x)/w(2);
y2 = -(1 + b + w(1)*x)/w(2);
y3 = -(b + w(1)*x - 1)/w(2);

% plot(x, y1);
% plot(x, y2);
% plot(x, y3);
hold on;

%-------Pinball loss SVM classifier--------

t = 0.5; %tau in pinball loss function
A_p = [-Y -X.*(Y*[1 1]) -eye(2*n); Y X.*(Y*[1 1]) -(1/t)*eye(2*n)];
B_p = [-ones(2*n, 1); ones(2*n, 1)];

[W_p, fval_p] = quadprog(H,f,A_p,B_p);
b_p = W_p(1, :);
w_p = W_p(2:3, :);

woptimal=[woptimal w_p];
boptimal=[boptimal b_p];

disp(b_p);
disp(w_p);

disp('-w(1)/w(2) for Pin-SVM:');
disp(-(w_p(1)/w_p(2)));

disp('-b/w(2) for Pin-SVM:');
disp(-(b_p/w_p(2)));

%--------Plotting the optimal Pinball loss hyperplane--------

y4 = -(b_p + w_p(1)*x)/w_p(2);
y5 = -(1 + b_p + w_p(1)*x)/w_p(2);
y6 = -(b_p + w_p(1)*x - 1)/w_p(2);

plot(x, y4,'.');
hold on;
plot(x, y5,'.');
hold on;
plot(x, y6,'.');
hold on;


% Insensitive(z) Pinball loss classifier

z = 0.2; %epsilon in insensitive zone
r=z/t;

A_in = [-Y -X.*(Y*[1 1]) -eye(2*n); Y X.*(Y*[1 1]) -(1/t)*eye(2*n)];
B_in = [-ones(2*n, 1)+z; ones(2*n, 1)+r];


[W_in, fval_in] = quadprog(H, f,A_in, B_in,Aeq,Beq,lb,ub);
b_in = W_in(1, :);
w_in = W_in(2:3, :);

woptimal=[woptimal w_in]
boptimal=[boptimal b_in]

disp(b_in);
disp(w_in);

disp('-w(1)/w(2) for Insensitive Pin-SVM:');
disp(-(w_in(1)/w_in(2)));

disp('-b/w(2) for Insensitive Pin-SVM:');
disp(-(b_in/w_in(2)));


%--------Plotting the optimal Insensitive Pinball loss hyperplane--------

y7 = -(b_in + w_in(1)*x)/w_in(2);
y8 = -(1 + b_in + w_in(1)*x)/w_in(2);
y9 = -(b_in + w_in(1)*x - 1)/w_in(2);

plot(x, y7,'*','MarkerIndices',1:1:length(y7));
hold on;
plot(x, y8,'*','MarkerIndices',1:1:length(y8));
hold on;
plot(x, y9,'*','MarkerIndices',1:1:length(y9));
hold off;




le=["Parameter ", "        PIN_SVM .  ", "       Hinge_Loss_SVM.  ", "    Insensitive Pin_SVM " ,"   Noise_level     "," Tau(t)   ", "    Epsilon(z) "];


pp=[10 20 30];
b_parameter=strings(0);
b_parameter(end+1)='b';
b_parameter(end+1)=num2str(boptimal(2));
b_parameter(end+1)=num2str(boptimal(1));
b_parameter(end+1)=num2str(boptimal(3));
b_parameter(end+1)=s;
b_parameter(end+1)=num2str(t);
b_parameter(end+1)=num2str(z);

w_parameter=strings(0);
w_parameter(end+1)="w";
str=strcat("(",num2str(woptimal(1,2)));str=strcat(str," ");str=strcat(str,num2str(woptimal(2,2)));str=strcat(str,")");
w_parameter(end+1)=str;
str=strcat("(",num2str(woptimal(1,1)));str=strcat(str," ");str=strcat(str,num2str(woptimal(2,1)));str=strcat(str,")");
w_parameter(end+1)=str;
str=strcat("(",num2str(woptimal(1,3)));str=strcat(str," ");str=strcat(str,num2str(woptimal(2,3)));str=strcat(str,")");
w_parameter(end+1)=str;
% w_parameter(end+1)=s;
% w_parameter(end+1)=num2str(t);
% w_parameter(end+1)=num2str(z);

slope=strings(0);
slope(end+1)='-w(1)/(w(2))';
slope(end+1)=num2str(-(w_p(1)/w_p(2)));
slope(end+1)=num2str(-(w(1)/w(2)));
slope(end+1)=num2str(-(w_in(1)/w_in(2)) );
% slope(end+1)=s;
% slope(end+1)=num2str(t);
% slope(end+1)=num2str(z);

b_const=strings(0);
b_const(end+1)='-b/(w(2))';
b_const(end+1)=num2str(-(b_p/w_p(2)));
b_const(end+1)=num2str(-(b/w(2)));
b_const(end+1)=num2str(-(b_in/w_in(2)));
% b_const(end+1)=s;
% b_const(end+1)=num2str(t);
% b_const(end+1)=num2str(z);

% Useful code.
% fid = fopen('lol.csv', 'w') ;
% fprintf(fid, '%s;', le{1,1:end-1}) ;
% fprintf(fid, '%s\n', le{1,end}) ;
% fclose(fid) ;

fid = fopen('lol.csv', 'a') ;
fprintf(fid, '%s;', b_parameter{1,1:end-1}) ;
fprintf(fid, '%s\n', b_parameter{1,end}) ;
fclose(fid) ;
% 
fid = fopen('lol.csv', 'a') ;
fprintf(fid, '%s;', w_parameter{1,1:end-1}) ;
fprintf(fid, '%s\n', w_parameter{1,end}) ;
fclose(fid) ;
% 
fid = fopen('lol.csv', 'a') ;
fprintf(fid, '%s;', slope{1,1:end-1}) ;
fprintf(fid, '%s\n', slope{1,end}) ;
fclose(fid) ;
% 
fid = fopen('lol.csv', 'a') ;
fprintf(fid, '%s;', b_const{1,1:end-1}) ;
fprintf(fid, '%s\n', b_const{1,end}) ;
fclose(fid) ;

%dlmwrite('lol.csv',str(1,:),'-append');

%
