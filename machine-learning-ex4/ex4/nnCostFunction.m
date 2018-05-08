function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

A1 = [ones(m,1), X]; % add column of 1's to X

% Initialize gradient matrices
Delta2 = zeros(num_labels, hidden_layer_size + 1);
Delta1 = zeros(hidden_layer_size, input_layer_size + 1);

% Iterate over all examples
for i=1:m

  % Forward propagate and compute costs
  a1 = A1(i, :); % input layer example

  z2 = a1 * transpose(Theta1);
  a2 = [1, sigmoid(z2)]; % hidden layer example value

  z3 = a2 * transpose(Theta2);
  h = a3 = sigmoid(z3); % output hypothesis example value

  yi = 1:num_labels == y(i); % label for example
  J += ((-1 .* yi * log(h') - (1 .- yi) * log(1 .- h'))); % increase J by cost for current example

  % Back Propagate: Calculate error terms
  d3 = (a3 - yi)';
  d2 = Theta2(:,2:end)' * d3 .* sigmoidGradient(z2');

  % fprintf('size d3: \n');
  % size(d3)
  % fprintf('size a2: \n');
  % size(a2)
  % fprintf('size d3 * a2: \n');
  % size(d3 * a2)
  %
  % fprintf('size d2: \n');
  % size(d2)
  % fprintf('size a1: \n');
  % size(a1)
  % fprintf('size d2 * a1: \n');
  % size(d2 * a1)

  % Back Propagate: Accumulate gradients
  Delta2 = Delta2 + d3 * a2;
  Delta1 = Delta1 + d2 * a1;

endfor

Theta2_grad = (Delta2 ./ m) + ...
              (lambda / m) * [zeros(num_labels,1),Theta2(:,2:end)];
Theta1_grad = Delta1 ./ m + ...
              (lambda / m) * [zeros(hidden_layer_size,1),Theta1(:,2:end)];

J /= m; % average J across examples

% add regularization term to J.
J += (lambda / (2 * m)) * (sum(Theta1(:,2:end)(:) .^ 2) + sum(Theta2(:,2:end)(:) .^ 2));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
