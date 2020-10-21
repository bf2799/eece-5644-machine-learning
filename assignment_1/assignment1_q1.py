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


# PART A ------------------------------------------------------

# Calculate likelihood ratios of all samples using knowledge of true data pdf. Then, generate ROC curve
likelihood_ratios_true_pdf = divide(multivariate_gaussian_pdf(samples_q1, m1, cov1),
                                    multivariate_gaussian_pdf(samples_q1, m0, cov0))
generate_roc_curve(likelihood_ratios_true_pdf, False, sample_labels_q1, p0, p1)

# PART B ----------------------------------------------------------------------

# Calculate likelihood ratios of all samples using Naive Bayesian Classifier. Then, generate ROC curve
likelihood_ratios_naive_bayesian = divide(multivariate_gaussian_pdf(samples_q1, m1, diag(diag(cov0))),
                                          multivariate_gaussian_pdf(samples_q1, m0, diag(diag(cov1))))
generate_roc_curve(likelihood_ratios_naive_bayesian, False, sample_labels_q1, p0, p1)

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
generate_roc_curve(array(likelihood_ratios_lda), True, sample_labels_q1, p0, p1)
