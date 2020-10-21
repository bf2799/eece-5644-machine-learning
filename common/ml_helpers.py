from numpy import *
import matplotlib.pyplot as plt


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


def multivariate_gaussian_pdf(x, mean, covariance):
    ret_matrix = []
    dimensions = len(mean)
    normalization_constant = ((2 * math.pi) ** (-dimensions / 2)) * (linalg.det(covariance) ** -0.5)
    cov_inv = linalg.inv(covariance)
    for i in range(len(x)):
        mean_diff = subtract(x[i], mean)
        exponent = math.exp(matmul(matmul(-0.5 * transpose(mean_diff), cov_inv), mean_diff))
        likelihood = normalization_constant * exponent
        ret_matrix.append(likelihood)
    return ret_matrix


def generate_roc_curve(likelihood_ratios_array, is_low_ratio_origin, sample_labels_array, prior_class0, prior_class1):
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
        cur_error = false_positives[i] * prior_class0 + false_negatives[i] * prior_class1
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