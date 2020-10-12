from numpy import *


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
