import ml_helpers as ml
import numpy as np
import matplotlib.pyplot as plt

RESTART_EXPERIMENTS = False
bic_performance_experiments = 'bic_performances.npy'
bic_gaussian_choices_experiments = 'bic_gaussian_choices.npy'
kfold_performance_experiments = 'kfold_performances.npy'
kfold_gaussian_choices_experiments = 'kfold_gaussian_choices.npy'

# Generate 2D gaussian mixture model with equal weights, means on a line, and circular symmetry
num_gaussians = 10
priors_q2 = 0.1 * np.ones(num_gaussians)
means_q2 = np.zeros([num_gaussians, 2])
for i in range(num_gaussians):
    means_q2[i][0] = i * 2
cov_vals_q2 = [0.35, 0.4, 0.15, 0.05, 0.3, 0.1, 0.45, 0.5, 0.25, 0.2]
covs_q2 = np.array([[[cov_vals_q2[i], 0], [0, cov_vals_q2[i]]] for i in range(num_gaussians)])
d_train_sample_lengths = [100, 1000, 10000, 100000]


def generate_random_dist():
    train = [[] for _ in range(len(d_train_sample_lengths))]
    for training_set in range(len(train)):
        for _ in range(d_train_sample_lengths[training_set]):
            rand_gaussian = ml.random_class_index(priors_q2)
            train[training_set].append(np.random.multivariate_normal(means_q2[rand_gaussian], covs_q2[rand_gaussian]))
    train = np.array(train, dtype='object')
    return train


# Show distribution of largest sample count
temp = generate_random_dist()
plt.plot([temp[-1][i][0] for i in range(len(temp[-1]))], [temp[-1][i][1] for i in range(len(temp[-1]))], 'bo',
         markersize=1)
plt.xlim(-2, 20)
plt.ylim(-11, 11)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Calculate BIC for all gaussian mixtures up to the max for a number of experiments
MAX_BIC_GAUSSIANS = 12
NUM_BIC_EXPERIMENTS = 50
if RESTART_EXPERIMENTS:
    bic_performances = [[] for i in range(len(d_train_sample_lengths))]
    bic_gaussian_choices = [[] for i in range(len(d_train_sample_lengths))]
else:
    bic_performances = np.load(bic_performance_experiments)
    bic_gaussian_choices = np.load(bic_gaussian_choices_experiments)


for experiment in range(NUM_BIC_EXPERIMENTS):
    d_train = generate_random_dist()
    for train_set in range(d_train.shape[0]):
        gaussian_choice, performance = ml.calculate_bic(np.array(d_train[train_set]), MAX_BIC_GAUSSIANS, False)
        bic_gaussian_choices[train_set].append(gaussian_choice)
        bic_performances[train_set].append(performance)
        print("BIC Experiment " + str(experiment + 1) + "/" + str(NUM_BIC_EXPERIMENTS) + " complete for " +
              str(d_train_sample_lengths[train_set]) + " samples")
    np.save(bic_performance_experiments, bic_performances)
    np.save(bic_gaussian_choices_experiments, bic_gaussian_choices)


def report_model_order_selection_performance(performances, gaussian_choices, selection_type):
    # Plot for different numbers of training sets and experiments
    x_plot = [i + 1 for i in range(len(performances[0][0]))]
    for training_set in range(len(performances)):
        # Scatter plot of average performance for each number of gaussians
        for exper in range(len(performances[0])):
            plt.scatter(x_plot, performances[training_set][exper], s=6)
        plt.plot(x_plot, np.mean(performances[training_set], axis=0))
        plt.legend(['Average ' + selection_type + ' Performance'])
        plt.title(selection_type + ' Performance for ' + str(d_train_sample_lengths[training_set]) + ' Samples')
        plt.xlabel('Gaussian Components (#)')
        plt.ylabel(selection_type + ' Performance')
        plt.show()
        # Histogram of chosen gaussian components
        plt.hist(gaussian_choices[training_set], bins=x_plot, align='left', rwidth=0.75)
        plt.title(selection_type + ' Selection for ' + str(d_train_sample_lengths[training_set]) + ' Samples')
        plt.xlabel('Gaussian Components (#)')
        plt.ylabel('Experiments Choosing (#)')
        plt.show()
    # Scatter plot of chosen gaussian components over number of samples
    for training_set in range(len(gaussian_choices)):
        plt.scatter([d_train_sample_lengths[training_set]] * len(gaussian_choices[0]), gaussian_choices[training_set],
                    s=6)
    print("Means: " + str(np.mean(gaussian_choices, axis=1)))
    plt.plot(d_train_sample_lengths, np.mean(gaussian_choices, axis=1))
    plt.plot(d_train_sample_lengths, [np.max(gaussian_choices[t_set]) for t_set in range(len(gaussian_choices))])
    plt.plot(d_train_sample_lengths, [np.min(gaussian_choices[t_set]) for t_set in range(len(gaussian_choices))])
    plt.xscale('log')
    plt.legend(['Average Chosen Gaussian Components', 'Max Chosen Gaussian Components',
                'Min Chosen Gaussian Components'])
    plt.title('Estimated Gaussian Components by Sample Number')
    plt.xlabel('Samples (#)')
    plt.ylabel('Gaussian Components (#)')
    plt.show()


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


K = 5
MAX_K_FOLD_GAUSSIANS = 12
NUM_K_FOLD_EXPERIMENTS = 50
if RESTART_EXPERIMENTS:
    k_fold_performances = [[] for i in range(len(d_train_sample_lengths))]
    k_fold_gaussian_choices = [[] for i in range(len(d_train_sample_lengths))]
else:
    k_fold_performances = np.load(kfold_performance_experiments)
    k_fold_gaussian_choices = np.load(kfold_gaussian_choices_experiments)

# Perform K-fold cross validation for each experiment and training set
for experiment in range(NUM_K_FOLD_EXPERIMENTS):
    d_train = generate_random_dist()
    for train_set in range(d_train.shape[0]):
        gaussian_choice, performance = k_fold_cross_validation(np.array(d_train[train_set]), K,
                                                                  MAX_K_FOLD_GAUSSIANS, False)
        k_fold_gaussian_choices[train_set].append(gaussian_choice)
        k_fold_performances[train_set].append(performance)
        print("K Fold Experiment " + str(experiment + 1) + "/" + str(NUM_K_FOLD_EXPERIMENTS) + " complete for " +
              str(d_train_sample_lengths[train_set]) + " samples")
    np.save(kfold_performance_experiments, k_fold_performances)
    np.save(kfold_gaussian_choices_experiments, k_fold_gaussian_choices)

# Report performance of model order selection for BIC
report_model_order_selection_performance(bic_performances, bic_gaussian_choices, 'BIC')
# Report performance of model order selection for K-fold cross validation
report_model_order_selection_performance(k_fold_performances, k_fold_gaussian_choices, 'K-Fold Cross Validation')
