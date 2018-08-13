mu = [0 0];
sigma = [1 -0.8; -0.8 1];

n = 90;

noise = mvnrnd(mu, sigma, n);
disp(noise);
Y_noise = randi([0, 1], n, 1)*2 - 1;
disp(Y_noise);

save('noise.mat', 'noise', 'Y_noise');
