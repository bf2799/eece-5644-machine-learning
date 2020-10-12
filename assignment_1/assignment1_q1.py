import matplotlib.pyplot as plt
from ml_helpers import *

NUM_GENERATED_SAMPLES = 10000

rand_gen = random.default_rng()
samples_q1 = []
sample_labels_q1 = []

# Class distributions
p0 = 0.7  # Prior probability for class 0
m0 = [-1, -1, -1, -1]  # Mean vector for class 0
cov0 = [[2, -0.5, 0.3, 0],  # Covariance matrix for class 0
        [-0.5, 1, -0.5, 0],
        [0.3, -0.5, 1, 0],
        [0, 0, 0, 2]]
p1 = 0.3  # Prior probability for class 1
m1 = [1, 1, 1, 1]  # Mean vector for class 1
cov1 = [[1, 0.3, -0.2, 0],  # Covariance matrix for class 1
        [0.3, 2, 0.3, 0],
        [-0.2, 0.3, 1, 0],
        [0, 0, 0, 3]]

# Generate samples from distributions
for index in range(NUM_GENERATED_SAMPLES):
    class_label = 0 if random.rand() < p0 else 1
    sample_labels_q1.append(class_label)
    if class_label == 0:
        samples_q1.append(random.multivariate_normal(m0, cov0))
    else:
        samples_q1.append(random.multivariate_normal(m1, cov1))


def generate_roc_curve(likelihood_ratios_array, is_low_ratio_origin, sample_labels_array):
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
    for i in range(NUM_GENERATED_SAMPLES):
        # Find all true/false positives/negatives
        if i == 0:
            true_positives.append(sum(sample_labels))
            false_positives.append(NUM_GENERATED_SAMPLES - true_positives[0])
            true_negatives.append(0)
            false_negatives.append(0)
            gammas.append(likelihood_ratios[i] - 1)  # Amount under lowest likelihood isn't important

        # Calculate gamma threshold for this iteration
        if i == NUM_GENERATED_SAMPLES - 1:
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
    for i in range(NUM_GENERATED_SAMPLES + 1):
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
    for i in range(NUM_GENERATED_SAMPLES):
        cur_error = false_positives[i] * p0 + false_negatives[i] * p1
        if cur_error < min_error_prob:
            min_error_prob = cur_error
            min_error_index = i
    print("P(error) = " + str(min_error_prob) + ", Gamma = " + str(gammas[min_error_index]))

    # Find area under ROC curve
    area = 0
    for i in range(1, NUM_GENERATED_SAMPLES):
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


# PART A ------------------------------------------------------

# Calculate likelihood ratios of all samples using knowledge of true data pdf. Then, generate ROC curve
likelihood_ratios_true_pdf = divide(multivariate_gaussian_pdf(samples_q1, m1, cov1),
                                    multivariate_gaussian_pdf(samples_q1, m0, cov0))
generate_roc_curve(likelihood_ratios_true_pdf, False, sample_labels_q1)

# PART B ----------------------------------------------------------------------

# Calculate likelihood ratios of all samples using Naive Bayesian Classifier. Then, generate ROC curve
likelihood_ratios_naive_bayesian = divide(multivariate_gaussian_pdf(samples_q1, m1, diag(diag(cov0))),
                                          multivariate_gaussian_pdf(samples_q1, m0, diag(diag(cov1))))
generate_roc_curve(likelihood_ratios_naive_bayesian, False, sample_labels_q1)

# PART C -----------------------------------------------------------------------

# Sort samples into class 0 and class 1 by label to prep for estimating mean and covariance
samples_q1_class0 = []
samples_q1_class1 = []
for j in range(NUM_GENERATED_SAMPLES):
    if sample_labels_q1[j] == 0:
        samples_q1_class0.append(samples_q1[j])
    else:
        samples_q1_class1.append(samples_q1[j])

# Estimate mean and covariance for class 0 and class 1 given samples
m0_hat = mean(samples_q1_class0, axis=0)
print("Class 0 Estimated Mean: \n" + str(m0_hat))
cov0_hat = cov(transpose(samples_q1_class0))
print("Class 0 Estimated Covariance: \n" + str(cov0_hat))
m1_hat = mean(samples_q1_class1, axis=0)
print("Class 1 Estimated Mean: \n" + str(m1_hat))
cov1_hat = cov(transpose(samples_q1_class1))
print("Class 1 Estimated Covariance: \n" + str(cov1_hat))

# Calculate between-class and within-class scatter
sb = matmul((m0_hat - m1_hat), transpose(m0_hat - m1_hat))
sw = cov0_hat + cov1_hat

# Find Fisher LDA Projection weight vector, which is the generalized eigenvector of (Sb, Sw) with the largest eigenvalue
eigenvalues_lda, eigenvectors_lda = linalg.eig(linalg.inv(sw) * sb)
eigenvalues_lda_sort_results = argsort(eigenvalues_lda)
w_lda = eigenvectors_lda[:, eigenvalues_lda_sort_results[-1]]
print("Fisher LDA Weight Vector: \n" + str(w_lda))

# Find likelihood ratios for Fisher LDA using transpose(w)*x. Then, generate ROC curve
likelihood_ratios_lda = []
for j in range(NUM_GENERATED_SAMPLES):
    likelihood_ratios_lda.append(matmul(transpose(w_lda), samples_q1[j]))
generate_roc_curve(array(likelihood_ratios_lda), True, sample_labels_q1)
