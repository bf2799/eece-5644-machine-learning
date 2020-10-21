import ml_helpers as ml
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp_optimize

# Class distributions
p0 = 0.6  # Prior probability for class 0
m01 = [5, 0]  # Mean vector for class 0 gaussian 1
cov01 = [[4, 0],  # Covariance matrix for class 0 gaussian 1
         [0, 2]]
w01 = 0.5  # Multiplier weight of guassian 1 to class 0 distribution
m02 = [0, 4]  # Mean vector for class 0 gaussian 2
cov02 = [[1, 0],  # Covariance matrix for class 0 gaussian 2
         [0, 3]]
w02 = 0.5  # Multiplier weight of guassian 2 to class 0 distribution
p1 = 0.4  # Prior probability for class 1
m1 = [3, 2]  # Mean vector for class 1
cov1 = [[2, 0],  # Covariance matrix for class 1
        [0, 2]]

# Generate samples from distributions
d_train_100 = []
d_train_100_labels = []
d_train_1000 = []
d_train_1000_labels = []
d_train_10000 = []
d_train_10000_labels = []
d_validate_20000 = []
d_validate_20000_labels = []
# Generate d_train_100
for index in range(100):
    class_label = 0 if np.random.rand() < p0 else 1
    d_train_100_labels.append(class_label)
    if class_label == 0:
        # Generate from gaussian distribution mixture
        class_0_gaussian = 1 if np.random.rand() < w01 else 2
        if class_0_gaussian == 1:
            d_train_100.append(np.random.multivariate_normal(m01, cov01))
        else:
            d_train_100.append(np.random.multivariate_normal(m02, cov02))
    else:
        d_train_100.append(np.random.multivariate_normal(m1, cov1))
# Generate d_train_1000
for index in range(1000):
    class_label = 0 if np.random.rand() < p0 else 1
    d_train_1000_labels.append(class_label)
    if class_label == 0:
        # Generate from gaussian distribution mixture
        class_0_gaussian = 1 if np.random.rand() < w01 else 2
        if class_0_gaussian == 1:
            d_train_1000.append(np.random.multivariate_normal(m01, cov01))
        else:
            d_train_1000.append(np.random.multivariate_normal(m02, cov02))
    else:
        d_train_1000.append(np.random.multivariate_normal(m1, cov1))
# Generate d_train_10000
for index in range(10000):
    class_label = 0 if np.random.rand() < p0 else 1
    d_train_10000_labels.append(class_label)
    if class_label == 0:
        # Generate from gaussian distribution mixture
        class_0_gaussian = 1 if np.random.rand() < w01 else 2
        if class_0_gaussian == 1:
            d_train_10000.append(np.random.multivariate_normal(m01, cov01))
        else:
            d_train_10000.append(np.random.multivariate_normal(m02, cov02))
    else:
        d_train_10000.append(np.random.multivariate_normal(m1, cov1))
# Generate d_validate_20000
for index in range(20000):
    class_label = 0 if np.random.rand() < p0 else 1
    d_validate_20000_labels.append(class_label)
    if class_label == 0:
        # Generate from gaussian distribution mixture
        class_0_gaussian = 1 if np.random.rand() < w01 else 2
        if class_0_gaussian == 1:
            d_validate_20000.append(np.random.multivariate_normal(m01, cov01))
        else:
            d_validate_20000.append(np.random.multivariate_normal(m02, cov02))
    else:
        d_validate_20000.append(np.random.multivariate_normal(m1, cov1))
d_train_100 = np.array(d_train_100)
d_train_1000 = np.array(d_train_1000)
d_train_10000 = np.array(d_train_10000)
d_validate_20000 = np.array(d_validate_20000)

# Plot validation data set (largest) to show sample distributions
class_0_samples = []
class_1_samples = []
for index in range(20000):
    if d_validate_20000_labels[index] == 0:
        class_0_samples.append(d_validate_20000[index])
    else:
        class_1_samples.append(d_validate_20000[index])
plt.scatter([row[0] for row in class_0_samples], [row[1] for row in class_0_samples], s=1, c='blue')
plt.scatter([row[0] for row in class_1_samples], [row[1] for row in class_1_samples], s=1, c='red')
plt.title("Class Distributions of Validation Samples")
plt.legend(["Class 0", "Class 1"], bbox_to_anchor=(1.1, 1))
plt.show()


# PART 1 --------------------------------------------------------------------------


def plot_theoretical_boundary():
    xy_grid_num = 100
    x_grid = np.linspace(min(d_validate_20000[:, 0]), max(d_validate_20000[:, 0]), xy_grid_num)
    y_grid = np.linspace(min(d_validate_20000[:, 1]), max(d_validate_20000[:, 1]), xy_grid_num)
    xy_array = np.zeros([xy_grid_num * xy_grid_num, 2])
    for i in range(xy_grid_num):
        for j in range(xy_grid_num):
            xy_array[i * xy_grid_num + j][0] = x_grid[i]
            xy_array[i * xy_grid_num + j][1] = y_grid[j]
    likelihood_ratios_grid = np.divide(ml.multivariate_gaussian_pdf(xy_array, m1, cov1),
                                       np.multiply(w01, ml.multivariate_gaussian_pdf(xy_array, m01, cov01)) +
                                       np.multiply(w02, ml.multivariate_gaussian_pdf(xy_array, m02, cov02)))
    z_grid = np.zeros([xy_grid_num, xy_grid_num])
    for i in range(xy_grid_num):
        for j in range(xy_grid_num):
            z_grid[i][j] = likelihood_ratios_grid[j * xy_grid_num + i]
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    return plt.contour(x_grid, y_grid, z_grid, [p1 / p0], colors=['black'], linewidths=2)


def report_parameters(p0_est, p1_est, w01_est, m01_est, cov01_est, w02_est, m02_est, cov02_est, m1_est, cov1_est):
    # Print estimates of parameters
    print("p0_hat: %s, p1_hat: %s\nw01_hat: %s, m01_hat: %s, cov01_hat: %s\nw02_hat: %s, m02_hat: %s, "
          "cov02_hat: %s\nm1_hat: %s, cov1_hat: %s" %
          (str(p0_est), str(p1_est), str(w01_est), str(m01_est), str(cov01_est), str(w02_est), str(m02_est),
           str(cov02_est), str(m1_est), str(cov1_est)))
    # Calculate likelihood ratios of all samples using knowledge of pdf. Then, generate ROC curve
    likelihood_ratios = np.divide(ml.multivariate_gaussian_pdf(d_validate_20000, m1_est, cov1_est),
                                  np.multiply(w01_est, ml.multivariate_gaussian_pdf(d_validate_20000, m01_est,
                                                                                    cov01_est)) +
                                  np.multiply(w02_est, ml.multivariate_gaussian_pdf(d_validate_20000, m02_est,
                                                                                    cov02_est)))
    p_error, gamma_boundary = ml.generate_roc_curve(likelihood_ratios, False, d_validate_20000_labels, p1, p0)

    # Plot theoretical and estimated decision boundaries on top of validation data
    xy_grid_num = 100
    x_grid = np.linspace(min(d_validate_20000[:, 0]), max(d_validate_20000[:, 0]), xy_grid_num)
    y_grid = np.linspace(min(d_validate_20000[:, 1]), max(d_validate_20000[:, 1]), xy_grid_num)
    xy_array = np.zeros([xy_grid_num * xy_grid_num, 2])
    for i in range(xy_grid_num):
        for j in range(xy_grid_num):
            xy_array[i * xy_grid_num + j][0] = x_grid[i]
            xy_array[i * xy_grid_num + j][1] = y_grid[j]
    likelihood_ratios_grid = np.divide(ml.multivariate_gaussian_pdf(xy_array, m1_est, cov1_est),
                                       np.multiply(w01_est, ml.multivariate_gaussian_pdf(xy_array, m01_est,
                                                                                         cov01_est)) +
                                       np.multiply(w02_est, ml.multivariate_gaussian_pdf(xy_array, m02_est, cov02_est)))
    z_grid = np.zeros([xy_grid_num, xy_grid_num])
    for i in range(xy_grid_num):
        for j in range(xy_grid_num):
            z_grid[i][j] = likelihood_ratios_grid[j * xy_grid_num + i]
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    plt.scatter([row[0] for row in class_0_samples], [row[1] for row in class_0_samples], s=1, c='blue')
    plt.scatter([row[0] for row in class_1_samples], [row[1] for row in class_1_samples], s=1, c='red')
    plt.title("Boundary for Deciding Classes 0 (Blue) and 1 (Red)")
    contour_est = plt.contour(x_grid, y_grid, z_grid, [gamma_boundary], colors=['lime'], linewidths=2)
    contour_theoretical = plot_theoretical_boundary()
    contour_est_legend, _ = contour_est.legend_elements()
    contour_theoretical_legend, _ = contour_theoretical.legend_elements()
    plt.legend([contour_est_legend[0], contour_theoretical_legend[0]], ['Estimated Boundary', 'Theoretical Boundary'],
               bbox_to_anchor=(1.1, 1))
    plt.show()


report_parameters(p0, p1, w01, m01, cov01, w02, m02, cov02, m1, cov1)


# PART 2 --------------------------------------------------------------------------


def estimate_class_parameters_gmm(samples, sample_labels):
    """
    Estimates parameters of classes from samples for this specific question model (one gaussian and one 2-gaussian mix)
    :param samples: Samples to estimate class parameters from
    :param sample_labels:
    :return: Estimates of class parameters, order shown in return statement
    """
    samples_class0 = [samples[i] for i in range(len(samples)) if sample_labels[i] == 0]
    samples_class1 = [samples[i] for i in range(len(samples)) if sample_labels[i] == 1]
    # Estimate class priors by taking labels of class / total samples
    p0_estimate = len(samples_class0) / len(samples)
    p1_estimate = len(samples_class1) / len(samples)
    # Estimate class 0 means and covariances as gaussian mixture model of 2 gaussians
    w0_estimates, m0_estimates, cov0_estimates, _ = ml.gmm_estimate_parameters(samples_class0, 2, 5, 0.0001)
    # Estimate class 1 means and covariances. Can just take sample mean and covariances due to single gaussian model
    m1_estimate = np.mean(samples_class1, axis=0)
    cov1_estimate = np.cov(np.transpose(samples_class1))
    return p0_estimate, p1_estimate, w0_estimates[0], m0_estimates[0], cov0_estimates[0], w0_estimates[1], \
        m0_estimates[1], cov0_estimates[1], m1_estimate, cov1_estimate


# Estimate and report with 10000 training samples
p0_hat, p1_hat, w01_hat, m01_hat, cov01_hat, w02_hat, m02_hat, cov02_hat, m1_hat, cov1_hat = \
        estimate_class_parameters_gmm(d_train_10000, d_train_10000_labels)
report_parameters(p0_hat, p1_hat, w01_hat, m01_hat, cov01_hat, w02_hat, m02_hat, cov02_hat, m1_hat, cov1_hat)
# Estimate and report with 1000 training samples
p0_hat, p1_hat, w01_hat, m01_hat, cov01_hat, w02_hat, m02_hat, cov02_hat, m1_hat, cov1_hat = \
        estimate_class_parameters_gmm(d_train_1000, d_train_1000_labels)
report_parameters(p0_hat, p1_hat, w01_hat, m01_hat, cov01_hat, w02_hat, m02_hat, cov02_hat, m1_hat, cov1_hat)
# Estimate and report with 100 training samples
p0_hat, p1_hat, w01_hat, m01_hat, cov01_hat, w02_hat, m02_hat, cov02_hat, m1_hat, cov1_hat = \
        estimate_class_parameters_gmm(d_train_100, d_train_100_labels)
report_parameters(p0_hat, p1_hat, w01_hat, m01_hat, cov01_hat, w02_hat, m02_hat, cov02_hat, m1_hat, cov1_hat)

# PART 3 --------------------------------------------------------------------------


def report_logistic_classification_performance(model_params, fit_type):
    # Calculate and minimum error probability on validation data using model params
    if fit_type == 'linear':
        likelihood = [model_params[0] + model_params[1] * d_validate_20000[i][0] +
                      model_params[2] * d_validate_20000[i][1] for i in range(20000)]
    elif fit_type == 'quadratic':
        likelihood = [model_params[0] + model_params[1] * d_validate_20000[i][0] +
                      model_params[2] * d_validate_20000[i][1] +
                      model_params[3] * (d_validate_20000[i][0] ** 2) +
                      model_params[4] * d_validate_20000[i][0] * d_validate_20000[i][1] +
                      model_params[5] * (d_validate_20000[i][1] ** 2) for i in range(20000)]
    else:
        print("Unknown fit type")
        exit(-1)
        return
    decisions = [1 if likelihood[i] < 0.5 else 0 for i in range(20000)]
    num_errors = 0
    for i in range(20000):
        num_errors += 1 if decisions[i] != d_validate_20000_labels[i] else 0
    error_probability = num_errors / 20000.0
    print("P(error): " + str(error_probability))
    # Plot theoretical and estimated decision boundaries on top of validation data
    grid_edge = 100
    x_grid = np.linspace(min(d_validate_20000[:, 0]), max(d_validate_20000[:, 0]), grid_edge)
    y_grid = np.linspace(min(d_validate_20000[:, 1]), max(d_validate_20000[:, 1]), grid_edge)
    z_grid = np.zeros([grid_edge, grid_edge])
    for i in range(grid_edge):
        for j in range(grid_edge):
            if fit_type == 'linear':
                z_grid[j][i] = model_params[0] + model_params[1] * x_grid[i] + model_params[2] * y_grid[j]
            elif fit_type == 'quadratic':
                z_grid[j][i] = model_params[0] + model_params[1] * x_grid[i] + model_params[2] * y_grid[j] + \
                               model_params[3] * (x_grid[i] ** 2) + model_params[4] * x_grid[i] * y_grid[j] + \
                               model_params[5] * (y_grid[j] ** 2)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    plt.scatter([row[0] for row in class_0_samples], [row[1] for row in class_0_samples], s=1, c='blue')
    plt.scatter([row[0] for row in class_1_samples], [row[1] for row in class_1_samples], s=1, c='red')
    plt.title("Boundary for Deciding Classes 0 (Blue) and 1 (Red)")
    contour_est = plt.contour(x_grid, y_grid, z_grid, [0.5], colors=['lime'], linewidths=2)
    contour_theoretical = plot_theoretical_boundary()
    contour_est_legend, _ = contour_est.legend_elements()
    contour_theoretical_legend, _ = contour_theoretical.legend_elements()
    plt.legend([contour_est_legend[0], contour_theoretical_legend[0]], ['Estimated Boundary', 'Theoretical Boundary'],
               bbox_to_anchor=(1.1, 1))
    plt.show()


# Perform logistic linear classification and report
model_parameters = np.array([0, 0, 0])
model_parameters = ml.logistic_binary_classification(d_train_100, d_train_100_labels, model_parameters, 'linear')
report_logistic_classification_performance(model_parameters, 'linear')
model_parameters = ml.logistic_binary_classification(d_train_1000, d_train_1000_labels, model_parameters, 'linear')
report_logistic_classification_performance(model_parameters, 'linear')
model_parameters = ml.logistic_binary_classification(d_train_10000, d_train_10000_labels, model_parameters, 'linear')
report_logistic_classification_performance(model_parameters, 'linear')
# Perform logistic quadratic classification and report
model_parameters = np.array([0, 0, 0, 0, 0, 0])
model_parameters = ml.logistic_binary_classification(d_train_100, d_train_100_labels, model_parameters, 'quadratic')
report_logistic_classification_performance(model_parameters, 'quadratic')
model_parameters = ml.logistic_binary_classification(d_train_1000, d_train_1000_labels, model_parameters, 'quadratic')
report_logistic_classification_performance(model_parameters, 'quadratic')
model_parameters = ml.logistic_binary_classification(d_train_10000, d_train_10000_labels, model_parameters, 'quadratic')
report_logistic_classification_performance(model_parameters, 'quadratic')
