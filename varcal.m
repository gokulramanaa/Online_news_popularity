function var = varcal(S, K, m)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
total1 = 0;
total2 = 0;

for (i=1:K)
   total1 = total1 + S(i,i);
end

for (j = 1:m)
  total2 = total2 + S(j,j);
end

var = total1/total2;
% =========================================================================
end