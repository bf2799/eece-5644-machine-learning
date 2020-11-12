import numpy as np
import matplotlib.pyplot as plt
import ml_helpers as ml
import sklearn as skl
from operator import itemgetter

# 4-class, 3-dimensional data distribution definitions
priors_q1 = np.array([0.25, 0.25, 0.25, 0.25])
means_q1 = np.array([[1, 2, 3],
                     [4, 2, 2],
                     [2, 1, 3],
                     [3, 4, 2]])
covs_q1 = np.array([[[1.2,  0.45,   0.2],
                     [0.45, 1.2,    0.75],
                     [0.2,  0.75,   0.8]],
                    [[0.9,  0.5,    0.5],
                     [0.5,  0.9,    0.9],
                     [0.5,  0.9,    1.2]],
                    [[0.5,  0.4,    0.3],
                     [0.4,  1,      0.75],
                     [0.3,  0.75,   0.7]],
                    [[1,    0.3,    0.75],
                     [0.3,  1.2,    0.5],
                     [0.75, 0.5,    0.8]]])


def generate_test_data(num_samples, num_dimensions):
	train = np.zeros(shape=[num_samples, num_dimensions])
	train_labels = np.zeros(shape=[num_samples])
	for n_sample in range(num_samples):
		rand_class = ml.random_class_index(priors_q1)
		train_labels[n_sample] = rand_class
		rand_sample = np.random.multivariate_normal(means_q1[rand_class], covs_q1[rand_class])
		train[n_sample] = rand_sample
	return train, train_labels


# Generate training data and labels
d_train_100, d_train_100_labels = generate_test_data(100, means_q1.shape[1])
d_train_200, d_train_200_labels = generate_test_data(200, means_q1.shape[1])
d_train_500, d_train_500_labels = generate_test_data(500, means_q1.shape[1])
d_train_1000, d_train_1000_labels = generate_test_data(1000, means_q1.shape[1])
d_train_2000, d_train_2000_labels = generate_test_data(2000, means_q1.shape[1])
d_train_5000, d_train_5000_labels = generate_test_data(5000, means_q1.shape[1])
d_test_100000, d_test_100000_labels = generate_test_data(100000, means_q1.shape[1])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xs=d_train_5000[:, 0], ys=d_train_5000[:, 1], zs=d_train_5000[:, 2])
# plt.show()

# Apply theoretically optimal classifier to test data using true data pdf
sample_likelihoods = np.zeros(shape=[priors_q1.shape[0], 100000])
for class_num in range(priors_q1.shape[0]):
	sample_likelihoods[class_num] = np.multiply(
		priors_q1[class_num], ml.multivariate_gaussian_pdf(d_test_100000, means_q1[class_num], covs_q1[class_num]))
sample_decisions = np.zeros(shape=[100000])
for sample in range(100000):
	sample_decisions[sample] = max(enumerate(sample_likelihoods[:, sample]), key=itemgetter(1))[0]
error_count = 0.0
for i in range(100000):
	error_count += 0 if sample_decisions[i] == d_test_100000_labels[i] else 1
empirical_min_error = error_count / 100000
print("Empirical Min Error Probability: " + str(empirical_min_error))


def nn_performance(nn, d_validate, d_validate_labels):
	"""
	Get performance of neural net on validation data
	:param nn: Neural net to apply
	:param d_validate: Validation samples
	:param d_validate_labels: Labels of validation samples
	:return: Log likelihood of correctly classifying validation data
	"""
	d_validate_predictions = nn.predict(d_validate)
	correct_count = 0.0
	if d_validate.shape[0] < 10:
		print("Something is wrong")
	for n_sample in range(d_validate.shape[0]):
		correct_count += 1 if d_validate_predictions[n_sample] == d_validate_labels[n_sample] else 0
	return np.log(correct_count / d_validate.shape[0])


def nn_train_performance(d_train, d_validate, model_order, d_train_labels, d_validate_labels, args=()):
	"""
	Trains neural net on training data with model order perceptrons in 1 hidden layer
	Returns performance on validation data
	:param d_train: Training data
	:param d_validate: Validation data
	:param model_order: Number of perceptrons in single hidden layer
	:param d_train_labels: Class labels for training data
	:param d_validate_labels: Class labels for validation data
	:param args: Extra arguments: (b_return_nn)
	:return: log likelihood of correct validation classification using trained model (high performance is good)
	"""
	nn = skl.neural_network.MLPClassifier(hidden_layer_sizes=(model_order,), activation='relu', solver='adam',
	                                      alpha=1e-6, max_iter=5000, shuffle=True, tol=1e-4, verbose=False,
	                                      warm_start=False, early_stopping=False, n_iter_no_change=10)
	nn.fit(d_train, d_train_labels)
	log_likelihood = nn_performance(nn, d_validate, d_validate_labels)
	# Return log likelihood of correct classification (and neural net if applicable)
	if args[0] is True:
		return log_likelihood, nn
	else:
		return log_likelihood


def report_best_train(d_train, d_train_labels, k, n_inits, d_test, d_test_labels):
	# Use 10-fold cross validation for each training set to get number of perceptrons to use
	perceptrons = ml.k_fold_cross_validation(d_train, k, nn_train_performance, 3, True, d_train_labels, (False,))
	perceptrons = int(np.round_(perceptrons, decimals=0))

	# Train the data using the number of perceptrons found in cross validation
	# Train multiple times and take best performance
	best_train = None
	best_train_performance = -1000000  # Very small number
	for _ in range(n_inits):
		performance, neural_net = nn_train_performance(d_train, d_train, perceptrons, d_train_labels, d_train_labels,
		                                               (True,))
		if performance > best_train_performance:
			best_train_performance = performance
			best_train = neural_net

	# Apply MLP to test data and get error probability
	log_likelihood = nn_performance(best_train, d_test, d_test_labels)
	error_prob = 1 - np.exp(log_likelihood)
	print("Error Probability for " + str(d_train.shape[0]) + " samples: " + str(error_prob))
	return error_prob, perceptrons


K_FOLD = 10
NUM_TRAIN_INITS = 5
error_prob_100, perceptrons_100 = report_best_train(d_train_100, d_train_100_labels, K_FOLD, NUM_TRAIN_INITS,
                                                    d_test_100000, d_test_100000_labels)
error_prob_200, perceptrons_200 = report_best_train(d_train_200, d_train_200_labels, K_FOLD, NUM_TRAIN_INITS,
                                                    d_test_100000, d_test_100000_labels)
error_prob_500, perceptrons_500 = report_best_train(d_train_500, d_train_500_labels, K_FOLD, NUM_TRAIN_INITS,
                                                    d_test_100000, d_test_100000_labels)
error_prob_1000, perceptrons_1000 = report_best_train(d_train_1000, d_train_1000_labels, K_FOLD, NUM_TRAIN_INITS,
                                                      d_test_100000, d_test_100000_labels)
error_prob_2000, perceptrons_2000 = report_best_train(d_train_2000, d_train_2000_labels, K_FOLD, NUM_TRAIN_INITS,
                                                      d_test_100000, d_test_100000_labels)
error_prob_5000, perceptrons_5000 = report_best_train(d_train_5000, d_train_5000_labels, K_FOLD, NUM_TRAIN_INITS,
                                                      d_test_100000, d_test_100000_labels)

# Plot error probability over number of samples with theoretical minimum
error_probs = [error_prob_100, error_prob_200, error_prob_500, error_prob_1000, error_prob_2000, error_prob_5000]
n_samples = [100, 200, 500, 1000, 2000, 5000]
plt.plot(n_samples, error_probs, 'b')
plt.plot([n_samples[0], n_samples[-1]], [empirical_min_error] * 2, 'g')
plt.xscale('log')
plt.title('Minimum Classification Error on Test Data Using Various Training Sizes')
plt.xlabel('Training Data Samples (#)')
plt.ylabel('Minimum Classification Error on Test Data')
plt.legend(['Training Min Error', 'Theoretical Min Error Estimate'])
plt.show()

# Plot number of perceptrons chosen over number of samples
percepts = [perceptrons_100, perceptrons_200, perceptrons_500, perceptrons_1000, perceptrons_2000, perceptrons_5000]
plt.plot(n_samples, percepts, 'b')
plt.xscale('log')
plt.title('Number of Chosen Perceptrons by Training Sample Size')
plt.xlabel('Training Data Samples (#)')
plt.ylabel('Chosen Perceptrons (#)')
plt.show()
