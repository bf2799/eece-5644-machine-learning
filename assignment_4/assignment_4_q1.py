import numpy as np
import sklearn.neural_network as skl
import ml_helpers as ml
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def read_data_from_csv(file_name):
	"""
	Reads data from CSV in format (x_samples, y_samples)
	Outputs 2 numpy arrays for sample inputs and target outputs
	:param file_name: Name of file to read from
	:return: Numpy array of inputs, shape (num_samples, num_features) and array of outputs, shape (num_samples,)
	"""
	file = open(file_name, 'r').readlines()
	file = np.array([line[:-1].split(',') for line in file])
	new_file = []
	for line in range(len(file)):
		new_file.append([float(sample) for sample in file[line]])
	input_array = np.transpose(new_file[:-1])
	return input_array, np.transpose(new_file[-1])


# Read training and test data from CSV
train_inputs, train_targets = read_data_from_csv('train_data_q1.csv')
test_inputs, test_targets = read_data_from_csv('test_data_q1.csv')


def nn_mse(nn, d_validate_input, d_validate_target):
	"""
	Returns MSE of predicted vs target outputs for neural network
	:param nn: Neural network whose model should be applied
	:param d_validate_input: Validation data, shape (n_samples, n_features)
	:param d_validate_target: Validation data target outputs, shape (n_samples,)
	:return: Mean squared error of predictions from model
	"""
	predictions = nn.predict(d_validate_input)
	tot_squared_error = 0
	for sample in range(len(d_validate_target)):
		tot_squared_error += (d_validate_target[sample] - predictions[sample]) ** 2
	mse = tot_squared_error / len(d_validate_target)
	return mse


def nn_train_performance(d_train_input, d_validate_input, n_perceptrons, d_train_target, d_validate_target, args=()):
	"""
	Trains regression neural net on training data with given number of perceptrons.
	Returns -MSE of predicted vs target outputs as performance metric (so higher performance is better)
	:param d_train_input: Training data, shape (n_samples, n_features)
	:param d_validate_input: Validation data, shape (n_samples, n_features)
	:param n_perceptrons: Number of perceptrons to put in single hidden layer of neural network
	:param d_train_target: Training data target outputs, shape (n_samples,)
	:param d_validate_target: Validation data target outputs, shape (n_samples,)
	:param args: Extra arguments: (b_return_nn) where
			b_return_nn: Boolean whether to return neural network object along with performance
	:return: Negative mean squared error and, if b_return_nn True, neural network object (skl.MLPRegressor)
	"""
	nn = skl.MLPRegressor(hidden_layer_sizes=(n_perceptrons,), activation='logistic', solver='adam', alpha=1e-6,
	                      max_iter=50000, shuffle=True, tol=1e-5, verbose=False, n_iter_no_change=20)
	nn.fit(d_train_input, d_train_target)
	neg_mse = -nn_mse(nn, d_validate_input, d_validate_target)
	# Return negative MSE (and neural net if applicable)
	if args[0] is True:
		return neg_mse, nn
	else:
		return neg_mse


# Get number of perceptrons to use in final training using 10-fold cross validation
perceptrons, intermediate_perceptrons = \
	ml.k_fold_cross_validation(d_train=train_inputs, K=10, performance_func=nn_train_performance,
	                           stop_consec_decreases=5, b_verbose=True, d_train_labels=train_targets,
	                           initial_order=1, order_step=1, args=(False,), b_return_all_selections=True,
	                           stop_consec_no_improvement=10)
perceptrons = int(np.round_(perceptrons, decimals=0))  # Convert to nearest integer
intermediate_perceptrons = [int(np.round_(ip, decimals=0)) for ip in intermediate_perceptrons]

# Show histogram of chosen perceptrons in k-fold cross validation
plt.hist(x=intermediate_perceptrons, histtype='bar', align='left', rwidth=0.75,
         bins=range(min(intermediate_perceptrons), max(intermediate_perceptrons) + 2, 1))
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.title('Chosen Perceptrons During K-Fold Cross Validation')
plt.ylabel('Perceptrons (#)')
plt.xlabel('Times Chosen (#)')
plt.show()

# Train final model using all training data and print the MSE
neg_mse_final, nn_final = nn_train_performance(train_inputs, train_inputs, perceptrons, train_targets, train_targets,
                                               (True,))
mse_final_train = -neg_mse_final  # Account for fact that performance is negative MSE
print("MSE Train: " + str(mse_final_train))

# Apply final model to test data
test_predictions = nn_final.predict(test_inputs)
mse_final_test = nn_mse(nn_final, test_inputs, test_targets)
print("MSE Test: " + str(mse_final_test))

# Plot predictions and actual outputs vs input
plt.scatter(test_inputs, test_targets, s=1, c='r')
plt.scatter(test_inputs, test_predictions, s=1, c='b')
plt.legend(['Actual Outputs', 'Predicted Outputs'])
plt.title('Question 1 Predicted and Actual Outputs by Input Value')
plt.xlabel('Input Value')
plt.ylabel('Output Value')
plt.show()
