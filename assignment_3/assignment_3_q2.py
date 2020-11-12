import numpy as np
import ml_helpers as ml
import matplotlib.pyplot as plt

# Problem constants
TRAIN_SAMPLES = 100
TEST_SAMPLES = 10000

# Problem data definitions
a = np.array([6, 5, 2, 4, 2, 1, 3])
x_mean = np.array([5, -3, 0, 1, -4, 2, -1])
x_cov = np.array([[1.5,  0.5,    0.2,    0.1,    0.3,    0.4,    0.3],
                  [0.5,  1.1,    0.6,    0.8,    0.2,    0.1,    0],
                  [0.2,  0.6,    1,      0.5,    0.2,    0,      0.9],
                  [0.1,  0.8,    0.5,    1,      0,      0.2,    0],
                  [0.3,  0.2,    0.2,    0,      0.7,    0,      0.5],
                  [0.4,  0.1,    0,      0.2,    0,      1,      0],
                  [0.3,  0,      0.9,    0,      0.5,    0,      1.5]])


def generate_data(num_samples, dimensions, alpha):
	data = []
	for sample in range(num_samples):
		x = np.random.multivariate_normal(x_mean, x_cov)
		z = np.random.multivariate_normal(np.zeros(dimensions), alpha * np.eye(dimensions))
		y = np.add(np.matmul(np.transpose(a), np.add(x, z)), np.random.normal(0, 1))
		data.append([x, y])
	data = np.array(data, dtype='object')
	return data


def weight_log_likelihood(weights, data, beta):
	"""
	Gets average log likelihood of weights occurring given data and assumed prior knowledge of weight vector
	:param weights: Weight vector for linear projection
	:param data: Data samples in form (x, y)
	:param beta: Scale factor for weight prior covariance identity matrix
	:return: Average log likelihood
	"""
	# Prior log likelihood = ln[(2πϐ)^(-weight len / 2)] - (w*w_transpose)/(2ϐ)
	prior_log_likelihood = np.log(2 * np.pi * beta) ** (-weights.shape[0] / 2) - \
	    np.dot(weights, np.transpose(weights)) / (2 * beta)
	# Sample log likelihood = ln[1/sqrt(2π)] - mean(1/2 * (w * [x_n, 1]_transpose - y_n)^2)
	sample_log_likelihood = np.log(1.0 / np.sqrt(2 * np.pi)) - \
	    np.mean([0.5 * (np.dot(weights, np.transpose(np.r_[sample[0], 1])) - sample[1]) ** 2 for sample in data])
	# Weight log likelihood = ln[p(data|w)] + ln[p(w)]
	return prior_log_likelihood + sample_log_likelihood


def weight_train_performance(d_train, d_validate, beta, args=()):
	"""
	Finds optimal weights with given beta as prior variance
	Returns performance of model on validation data
	:param d_train: Training data
	:param d_validate: Validation data
	:param beta: Weight prior variance around 0-mean vector
	:param args: Extra arguments (b_return_model)
	:return: Log likelihood of weights given samples and beta (high performance is good)
	"""
	# Weights based on derived MAP estimator
	w_hat = np.mean([np.array(sample[1] / (np.dot(np.r_[sample[0], 1], np.transpose(np.r_[sample[0], 1])) - 1.0 / beta)
	                          * np.r_[sample[0], 1]) for sample in d_train], axis=0)
	# Return negative log likelihood of optimization weight results using validation data
	log_likelihood = weight_log_likelihood(w_hat, d_validate, beta)
	if args[0] is True:
		return log_likelihood, w_hat
	else:
		return log_likelihood


def collect_weight_training(z_alpha):
	# Generate sets of training data
	d_train = generate_data(TRAIN_SAMPLES, x_mean.shape[0], z_alpha)
	d_test = generate_data(TEST_SAMPLES, x_mean.shape[0], z_alpha)
	# Choose a beta using 10-fold cross validation
	beta_guess = ml.k_fold_cross_validation(d_train=d_train, K=10, performance_func=weight_train_performance,
	                                        stop_consec_decreases=100, b_verbose=False, initial_order=0.00001,
	                                        order_step=0.00001, args=(False, ))
	# Find best weights using all training data
	_, best_weights = weight_train_performance(d_train, d_train, beta_guess, (True, ))
	# Find -2 log likelihood of weights on validation data
	neg2loglikelihood = -2 * weight_log_likelihood(best_weights, d_test, beta_guess)
	# Find MSE of weights from the MAP-expected 0
	weight_mse_from_zero = np.mean([best_weights[dim] ** 2 for dim in range(best_weights.shape[0])])
	return beta_guess, neg2loglikelihood, weight_mse_from_zero


# Get training results for various alphas
z_alphas = np.trace(x_cov) / x_mean.shape[0] * np.logspace(-3, 3, num=50)
training_results = []  # Each tuple inside should be (beta_guess, neg2loglikelihood)
for alph in z_alphas:
	training_results.append(collect_weight_training(alph))
	print("Z alpha complete: " + str(alph))
training_results = np.array(training_results)


def plot_solutions_vs_alpha(training_result):
	plt.scatter(z_alphas, training_result, c='b', s=5)
	plt.xlabel('Z Alpha')
	plt.xscale('log')
	plt.show()


# Plot beta at various alphas
plt.title('Betas vs Alphas')
plt.ylabel('Betas')
plot_solutions_vs_alpha(training_results[:, 0])

# Plot -2 log likelihood as alphas
plt.title('-2 Log Likelihood on Test Data vs Alphas')
plt.ylabel('-2 Log Likelihood')
plot_solutions_vs_alpha(training_results[:, 1])

# Plot mean squared error of weights from MAP-assumed values of 0
plt.title('MSE of Weights from MAP Expectation of 0')
plt.ylabel('MSE')
plot_solutions_vs_alpha(training_results[:, 2])