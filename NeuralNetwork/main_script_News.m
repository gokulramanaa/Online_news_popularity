%%Hand digit classifier using Neural Network 
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 35;  % 20x20 Input Images of Digits
hidden_layer_size = 10;   % 25 hidden units
num_labels = 1;          % 10 labels, from 1 to 10   
                          % (note that "0" mapped to label 10)

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')
pkg load io
data = csv2cell("train.csv");
X_train = data(:,1:59); X_train = cell2mat(X_train); y_train=data(:,60); y_norm = cell2mat(y_train);

[X_norm, mu, sigma] = featureNormalize(X_train);
[U, S] = pca(X_norm);
n = 59;
K = 2;
Z = projectData(X_norm, U, K);
X_rec  = recoverData(Z, U, K);

var = varcal(S,K,n)

hold on;
drawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2);
drawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2);
hold off;


%%%%%%%%%%hold on;
%%%%%%%%%%%drawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2);
%%%%%%%%%%%drawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2);
%%%%%%%%%%%%hold off;

%load('number_data.mat');
m = size(X_norm, 1);

% Randomly select 100 data points to display
sel = randperm(size(X_norm, 1));

% Splitting 90% of the data for training and 10% data for testing
X_train = Z(sel(1:36000),:);
y_train = y_norm(sel(1:36000),:);
X_test = Z(sel(36001:end),:);
y_test = y_norm(sel(36001:end),:);

%% Initializing Pameters 
fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% Implement Regularization

fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')
%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);
% Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X_train, y_train, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' ...
         '\n\n'], lambda, debug_J);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% Training NN 

fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', 50);

%lambda can be tried with different values
lambda = 1;

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

[nn_params, cost] = fmincg(costFunction, nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Implement Predict =================
pred = predict(Theta1, Theta2, X_train);
pred1 = predict(Theta1,Theta2,X_test);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_train)) * 100);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred1 == y_test)) * 100);



