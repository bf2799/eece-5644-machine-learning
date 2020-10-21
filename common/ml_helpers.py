from numpy import *
import matplotlib.pyplot as plt
from sklearn import *
from operator import itemgetter
import scipy.optimize as sp_optimize
import sys


def gaussian_minimum_expected_loss_decisions(samples, loss_matrix, class_means, class_covs,
                                             class_priors):
    # Find number of classes and validate that number
    num_classes = len(loss_matrix)
    if len(loss_matrix[0]) != num_classes or len(class_means) != num_classes or len(class_covs) != num_classes:
        print("Error: Non-matching number of classes passed to gaussian_minimum_expected_loss function")
        return
    # Loop through all samples to decide class for each
    sample_decisions = []
    for sample_index in range(len(samples)):
        # Loop through all classes to find risk associated with assigning sample to each
        risk = zeros([num_classes])
        for assumed_class in range(num_classes):
            total_loss = 0
            # Loop through all classes to calculate loss associated with deciding each given assumed class
            for decision_class in range(num_classes):
                total_loss += loss_matrix[assumed_class][decision_class] \
                              * class_priors[decision_class] \
                              * multivariate_gaussian_pdf([samples[sample_index]], class_means[decision_class],
                                                          class_covs[decision_class])[0]
            risk[assumed_class] = total_loss
        # Find minimum risk and add to sample decisions
        minimum_risk_class = argsort(risk)[0] + 1
        sample_decisions.append(minimum_risk_class)

    return sample_decisions


def generate_confusion_matrix_counts(sample_decisions, sample_labels, num_classes):
    # Validate that number of sample decisions and labels are the same
    if len(sample_decisions) != len(sample_labels):
        print("Error: Non-matching number of samples passed to generate_confusion_matrix function")
        return
    # Create confusion matrix with rows = predicted classes, columns = actual classes
    confusion_matrix_count = zeros([num_classes, num_classes])
    for sample_index in range(len(sample_decisions)):
        confusion_matrix_count[sample_decisions[sample_index] - 1][sample_labels[sample_index] - 1] += 1
    return confusion_matrix_count


def generate_confusion_matrix(sample_decisions, sample_labels, num_classes):
    confusion_matrix_count = generate_confusion_matrix_counts(sample_decisions, sample_labels, num_classes)
    # Create confusion matrix with rows = predicted classes, columns = actual classes
    confusion_matrix = zeros([num_classes, num_classes])
    for row in range(num_classes):
        for col in range(num_classes):
            confusion_matrix[row][col] = confusion_matrix_count[row][col] / sum(confusion_matrix_count[:, col])
    return confusion_matrix


def multivariate_gaussian_pdf(x, mean, covariance, x_len=-1):
    """
    Returns likelihoods of all samples in array given mean and covariance
    :param x: Array of samples
    :param mean: Mean of multivariate distribution as 1-D matrix
    :param covariance: Covariance of multivariate distribution as 2-D matrix
    :param x_len: Length of x, helps speed up algorithm when this is called a lot
    :return: Array of likelihoods
    """
    if x_len == -1:
        x_len = len(x)
    ret_matrix = []
    dimensions = len(mean)
    normalization_constant = ((2 * math.pi) ** (-dimensions / 2)) * (linalg.det(covariance) ** -0.5)
    cov_inv = linalg.inv(covariance)
    for i in range(x_len):
        mean_diff = subtract(x[i], mean)
        exponent = math.exp(matmul(matmul(-0.5 * transpose(mean_diff), cov_inv), mean_diff))
        likelihood = normalization_constant * exponent
        ret_matrix.append(likelihood)
    return ret_matrix


def generate_roc_curve(likelihood_ratios_array, is_low_ratio_origin, sample_labels_array, prior_class_denominator,
                       prior_class_numerator):
    # Check for valid array lengths
    if len(likelihood_ratios_array) != len(sample_labels_array):
        return

    # Sort likelihood ratios, samples, and labels to make selecting good gamma threshold values
    likelihood_sort_results = argsort(likelihood_ratios_array * (-1 if is_low_ratio_origin else 1))
    likelihood_ratios = likelihood_ratios_array[likelihood_sort_results]
    sample_labels = array(sample_labels_array)[likelihood_sort_results]

    # True/False Positives/Negatives numbers instead of percentages at first for more efficient looping through samples
    true_positives = []
    false_positives = []
    true_negatives = []
    false_negatives = []
    gammas = []
    # Keep looping to increase gamma threshold until all samples are classified into same class
    for i in range(len(likelihood_ratios_array)):
        # Find all true/false positives/negatives
        if i == 0:
            true_positives.append(sum(sample_labels))
            false_positives.append(len(likelihood_ratios_array) - true_positives[0])
            true_negatives.append(0)
            false_negatives.append(0)
            gammas.append(likelihood_ratios[i] - 1)  # Amount under lowest likelihood isn't important

        # Calculate gamma threshold for this iteration
        if i == len(likelihood_ratios_array) - 1:
            gamma_threshold = likelihood_ratios[i] + 1  # The amount over the highest likelihood isn't important
        else:
            gamma_threshold = (likelihood_ratios[i] + likelihood_ratios[i + 1]) / 2
        gammas.append(gamma_threshold)

        # Find which positive is subtracted from and which negative is added to based on label of likelihood passed
        temp_true_positives = 0
        temp_false_positives = 0
        temp_true_negatives = 0
        temp_false_negatives = 0
        if sample_labels[i] == 0:
            temp_false_positives = -1
            temp_true_negatives = 1
        else:
            temp_true_positives = -1
            temp_false_negatives = 1
        true_positives.append(true_positives[-1] + temp_true_positives)
        false_positives.append(false_positives[-1] + temp_false_positives)
        true_negatives.append(true_negatives[-1] + temp_true_negatives)
        false_negatives.append(false_negatives[-1] + temp_false_negatives)

    # Change true/false positives/negatives from numbers to percentages
    for i in range(len(likelihood_ratios_array) + 1):
        temp_tp = true_positives[i] / (true_positives[i] + false_negatives[i]) if (true_positives[i] +
                                                                                   false_negatives[i]) > 0 else 0
        temp_fp = false_positives[i] / (false_positives[i] + true_negatives[i]) if false_positives[i] + \
            true_negatives[i] > 0 else 0
        temp_tn = true_negatives[i] / (false_positives[i] + true_negatives[i]) if false_positives[i] + \
            true_negatives[i] > 0 else 0
        temp_fn = false_negatives[i] / (true_positives[i] + false_negatives[i]) if (true_positives[i] +
                                                                                    false_negatives[i]) > 0 else 0
        true_positives[i] = temp_tp
        false_positives[i] = temp_fp
        true_negatives[i] = temp_tn
        false_negatives[i] = temp_fn

    # Find minimum probability of error
    min_error_prob = 1
    min_error_index = 0
    for i in range(len(likelihood_ratios_array)):
        cur_error = false_positives[i] * prior_class_denominator + false_negatives[i] * prior_class_numerator
        if cur_error < min_error_prob:
            min_error_prob = cur_error
            min_error_index = i
    print("P(error) = " + str(min_error_prob) + ", Gamma = " + str(gammas[min_error_index]))

    # Find area under ROC curve
    area = 0
    for i in range(1, len(likelihood_ratios_array)):
        area += (true_positives[i] + true_positives[i - 1]) / 2 * (false_positives[i - 1] - false_positives[i])

    # Plot ROC curve and min probability of error
    plt.plot(false_positives, true_positives, 'b',
             false_positives[min_error_index], true_positives[min_error_index], 'ro')
    plt.title("Minimum Expected Risk ROC Curve")
    plt.xlabel("P (False Positive)")
    plt.ylabel("P (True Positive)")
    plt.legend(['ROC Curve', 'Estimated Min Error'])
    plt.text(0.4, 0.5, "Area Under ROC: " + str(round(area, 5)))
    plt.text(false_positives[min_error_index] + 0.03, true_positives[min_error_index] - 0.03,
             "(" + str(round(false_positives[min_error_index], 3)) + "," +
             str(round(true_positives[min_error_index], 3)) + ")")
    plt.show()

    return min_error_prob, gammas[min_error_index]


def gmm_estimate_parameters(samples, num_gaussians, num_inits, convergence_threshold):
    new_samples = samples
    sample_len = len(samples)
    max_log_likelihood = -10000000000000000 # Very very small number
    max_priors = [0] * num_gaussians
    max_means = [0] * num_gaussians
    max_covs = [0] * num_gaussians
    for loop in range(num_inits):
        random.shuffle(new_samples)
        # Initialize priors, means, and covariances
        # Priors initially all equal
        priors = [1 / num_gaussians] * num_gaussians
        # Means are means of samples split equally into classes
        means = [mean(new_samples[round(i * sample_len / num_gaussians):
                                  round((i + 1) * sample_len / num_gaussians - 1)],
                 axis=0) for i in range(num_gaussians)]
        # Covariances are covariances of samples split equally into classes
        covs = [cov(transpose(new_samples[round(i * sample_len / num_gaussians):
                                          round((i + 1) * sample_len / num_gaussians - 1)]))
                for i in range(num_gaussians)]
        converged = False
        # Keep iterating algorithm while it hasn't converged
        while not converged:
            # Class likelihoods given samples have rows = classes and columns = samples
            class_likelihoods_temp = [
                (multiply(priors[i], multivariate_gaussian_pdf(new_samples, means[i], covs[i], sample_len)))
                for i in range(num_gaussians)]
            class_likelihoods_temp_column_sums = sum(class_likelihoods_temp, axis=0)
            class_likelihoods_given_samples = [[class_likelihoods_temp[i][j] / class_likelihoods_temp_column_sums[j]
                                                for j in range(sample_len)] for i in range(num_gaussians)]
            # Calculate new priors, means, and covariance values
            priors_new = [mean(class_likelihoods_given_samples[i]) for i in range(num_gaussians)]
            means_new = [divide(sum([multiply(new_samples[j], class_likelihoods_given_samples[i][j])
                                     for j in range(sample_len)], axis=0),
                                sum(class_likelihoods_given_samples[i]))
                         for i in range(num_gaussians)]
            covs_new = [add(divide(sum([multiply(class_likelihoods_given_samples[i][j],
                                                 outer((subtract(new_samples[j], means_new[i])),
                                                       transpose(subtract(new_samples[j], means_new[i]))))
                                        for j in range(sample_len)], axis=0),
                                   sum(class_likelihoods_given_samples[i])), 0.0000000001 * identity(len(samples[0])))
                        for i in range(num_gaussians)]
            # Check for convergence
            if mean(absolute(subtract(priors_new, priors))) + \
                mean(absolute(subtract(means_new, means))) + \
                mean(absolute(subtract(covs_new, covs))) < convergence_threshold:
                converged = True
            # Set new prior, mean, and covariance values
            priors = priors_new
            means = means_new
            covs = covs_new

        # Use estimated parameters to find likelihood. Save if this is the best likelihood of all initializations
        pdfs = zeros(sample_len)
        for i in range(num_gaussians):
            temp_pdf = add(pdfs, multiply(priors[i],
                                          multivariate_gaussian_pdf(new_samples, means[i], covs[i])))
            pdfs = temp_pdf
        log_likelihood = sum(log(pdfs))
        if log_likelihood > max_log_likelihood:
            max_log_likelihood = log_likelihood
            max_priors = priors
            max_means = means
            max_covs = covs

    return max_priors, max_means, max_covs, max_log_likelihood


def random_gaussian_class_index(priors):
    """
    Returns a weighted random index from 0 to len(priors) (uninclusive) based on the prior values
    :param priors: Class priors that must add up to 1
    :return: Index from 0 to len(priors)-1
    """
    rand_num = random.rand()
    summer = 0
    for j in range(len(priors)):
        summer += priors[j]
        if rand_num < summer:
            return j
    return -1  # Should never get here, but return something that will never happen if we do get here


def calculate_bic(d_train, max_gaussians, b_verbose):
    """
    Calculates BIC and returns array of BIC score at each gaussian
    :param d_train: Training data as numpy array
    :param max_gaussians: Max number of gaussians to calculate BIC for
    :param b_verbose: Boolean to enable/disable printing of progress
    :return: Chosen num gaussians, numpy array of BIC values from 1 to max gaussians
    """
    # Calculate BIC model-order criterion
    bic_array = []
    for gaussians in range(1, max_gaussians + 1):
        dist = mixture.GaussianMixture(n_components=gaussians, covariance_type='diag', n_init=3,
                                       init_params='kmeans', max_iter=100000, tol=0.0001, reg_covar=1e-10)
        dist.fit(d_train)
        log_likelihood = sum(dist.score_samples(d_train))
        bic = -2 * log_likelihood + (gaussians * (1 + d_train.shape[1] * 2 +
                                                  sum([i for i in range(1, d_train.shape[1])])) -
                                     1) * log(d_train.shape[0])
        bic_array.append(bic)
        if b_verbose:
            print(str(len(d_train)) + "-sample BIC for " + str(gaussians) + " Classes: " + str(bic))
    min_index = min(enumerate(bic_array), key=itemgetter(1))[0]
    return min_index + 1, bic_array


def k_fold_cross_validation(d_train, K, max_gaussians, b_verbose):
    """
    Performs k-fold cross validation and returns array of average log likelihood at each number of gaussians
    :param d_train: Data to run k-fold cross validation on
    :param K: Number of partitions to split data into for training/validation
    :param max_gaussians: Number of gaussians to get performance at
    :param b_verbose: Whether to print intermittent statuses to console
    :return: Chosen num gaussians, numpy array of average log likelihood at each number of gaussians
    """
    # Get indices to partition data into K parts to prep for K-fold cross validation
    partition_indexes = r_[linspace(0, d_train.shape[0], num=K, endpoint=False, dtype=int), d_train.shape[0]]
    # Loop through using different data partition as validation data
    performances = [[] for i in range(K)]
    for k in range(K):
        # Get training and validation data sets for this iteration of k
        d_train_temp = r_[d_train[:partition_indexes[k]], d_train[partition_indexes[k + 1]:]]
        d_validate_temp = d_train[partition_indexes[k]:partition_indexes[k + 1]]
        # Increase # of gaussians in model until max gaussians reached
        for num_gaussians in range(1, max_gaussians + 1):
            # Create gaussian mixture and fit it to temp training data
            dist = mixture.GaussianMixture(n_components=num_gaussians, covariance_type='diag', n_init=3,
                                           init_params='kmeans', max_iter=100000, tol=0.0001, reg_covar=1e-10)
            dist.fit(d_train_temp)
            # Weighted log likelihood is average log likelihood of samples on validation data
            log_likelihood = dist.score(d_validate_temp)
            # Save performance
            performances[k].append(log_likelihood)
            if b_verbose:
                print(str(num_gaussians) + " gaussians for K " + str(k + 1) + "/" + str(K) + " and sample size " +
                      str(d_train.shape[0]) + " complete")
    # Return array of the average log likelihood at each number of gaussians and max performance
    mean_performances = mean(array(performances), axis=0)
    max_index = max(enumerate(mean_performances), key=itemgetter(1))[0]
    return max_index + 1, mean_performances.tolist()


def logistic_binary_classification_likelihood(model_params, d_train, d_train_labels, fit_type):
    """
    Calculates average negative log likelihood of class posteriors given sample x
    :param model_params: Vector of model parameters to fit sample to
    :param d_train: Training data
    :param d_train_labels: Label of training data
    :param fit_type: Type of fit to look for. Must be linear, quadratic
    :return: Average negative log likelihood of choosing correct class given sample
    """
    if fit_type == 'linear':
        z = [r_[1, sample] for sample in d_train]
    elif fit_type == 'quadratic':
        z = [r_[1, sample, sample[0] ** 2, sample[0] * sample[1], sample[1] ** 2] for sample in d_train]
    else:
        print('Logistic Binary Classification Unknown fit type')
        exit(-1)
        return
    # Logistic values are 1/(1+e^wz), where w is model params and z is sample weight vector
    logistic_values = [1.0 / (1 + exp(matmul(model_params, z[sample]))) for sample in range(len(d_train))]
    # Likelihood is 1 - logistic value if class = 0
    correct_class_likelihoods = [(1 - logistic_values[i] if d_train_labels[i] == 0 else logistic_values[i])
                                 for i in range(len(d_train))]
    # Average the log likelihoods of being the correct class
    return -mean(log(correct_class_likelihoods))


def logistic_binary_classification(d_train, d_train_labels, model_params_init, fit_type):
    """
    Performs logistic-based binary classification and returns model parameters
    :param d_train: Training data
    :param d_train_labels: Training data labels
    :param model_params_init: Initial estimates of model parameters
    :param fit_type: Type of fit to look for. Must be linear, quadratic
    :return:
    """
    # Find minimized logistic binary classification function and return if successful
    optimization_result = sp_optimize.minimize(fun=logistic_binary_classification_likelihood, x0=model_params_init,
                                               args=(d_train, d_train_labels, fit_type), method='Nelder-Mead',
                                               options={'maxiter': 5000, 'fatol': 0.001})
    if not optimization_result.success:
        print(optimization_result.message)
        exit(-1)
    return optimization_result.x