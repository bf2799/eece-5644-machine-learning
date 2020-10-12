from ml_helpers import *
import matplotlib.pyplot as plt

NUM_GENERATED_SAMPLES = 10000
rand_gen = random.default_rng()
samples_q2 = []
sample_labels_q2 = []

# Class distributions with means at corners of square
means = [[-1, 1, 0], 	# Class 1
		 [1, 1, 0],		# Class 2
		 [-1, -1, 0],	# Class 3
		 [1, -1, 0]]	# Class 4
covs = [[[0.4, 0, 0],	# Class 1
		[0, 0.4, 0],
		[0, 0, 0.4]],
		[[0.3, 0, 0],	# Class 2
		 [0, 0.3, 0],
		 [0, 0, 0.3]],
		[[0.2, 0, 0],	# Class 3
		 [0, 0.2, 0],
		 [0, 0, 0.2]],
		[[0.25, 0, 0],	# Class 4
		 [0, 0.25, 0],
		 [0, 0, 0.25]]]
priors = [0.2, 0.25, 0.25, 0.3]

# Generate samples from distributions
for i in range(NUM_GENERATED_SAMPLES):
	rand_prob = random.rand()
	if rand_prob < priors[0]:
		sample_labels_q2.append(1)
		samples_q2.append(random.multivariate_normal(means[0], covs[0]))
	elif rand_prob < priors[0] + priors[1]:
		sample_labels_q2.append(2)
		samples_q2.append(random.multivariate_normal(means[1], covs[1]))
	elif rand_prob < priors[0] + priors[1] + priors[2]:
		sample_labels_q2.append(3)
		samples_q2.append(random.multivariate_normal(means[2], covs[2]))
	else:
		sample_labels_q2.append(4)
		samples_q2.append(random.multivariate_normal(means[3], covs[3]))
samples_q2 = array(samples_q2)
sample_labels_q2 = array(sample_labels_q2)

# PART A ------------------------------------------------------------------------------------------------

# Use 0-1 loss to choose classes for samples
loss_matrix_a = [[0, 1, 1, 1],
				  [1, 0, 1, 1],
				  [1, 1, 0, 1],
				  [1, 1, 1, 0]]
sample_decisions_a = gaussian_minimum_expected_loss_decisions(samples_q2, loss_matrix_a, means, covs, priors)

# Create and print confusion matrix
confusion_matrix_a = generate_confusion_matrix(sample_decisions_a, sample_labels_q2, len(means))
print("0-1 Loss Confusion Matrix (Predicted Class on Rows)\n" + str(confusion_matrix_a))


def scatter_plot(samples, sample_labels, sample_decisions, num_classes, plot_title):
	# Array of samples, broken into classes then correct/incorrect predictions
	plot_arrays = []
	for i in range(num_classes):
		plot_arrays.append([[], []])
	for sample_index in range(len(sample_labels)):
		if sample_labels[sample_index] == sample_decisions[sample_index]:
			plot_arrays[sample_labels[sample_index] - 1][0].append(samples[sample_index])
		else:
			plot_arrays[sample_labels[sample_index] - 1][1].append(samples[sample_index])
	# Make 2-D scatterplot (x-y dimensions) with following class markers and green/red for correct/incorrect predictions
	class_plot_markers = ['.', 'o', '^', 's']
	for class_num in range(num_classes):
		plt.plot([row[0] for row in plot_arrays[class_num][0]], [row[1] for row in plot_arrays[class_num][0]],
				 'g' + class_plot_markers[class_num], markersize=3)
	for class_num in range(len(means)):
		plt.plot([row[0] for row in plot_arrays[class_num][1]], [row[1] for row in plot_arrays[class_num][1]],
				 'r' + class_plot_markers[class_num], markersize=3)
	plt.title(plot_title)
	legend_entries = []
	for entry in range(num_classes):
		legend_entries.append("Class " + str(entry + 1) + " Correct")
	for entry in range(num_classes):
		legend_entries.append("Class " + str(entry + 1) + " Incorrect")
	plt.legend(legend_entries, bbox_to_anchor=(1.02, 1))
	plt.show()


# Create 2D scatter plot for Part A
scatter_plot(samples_q2, sample_labels_q2, sample_decisions_a, len(means),
			 "0-1 Loss X-Y Scatterplot with Classes and Predictions")

# PART B ------------------------------------------------------------------------------------------------

# Use given loss to choose classes for samples
loss_matrix_b = [[0, 1, 2, 3],
				[10, 0, 5, 10],
				[20, 10, 0, 1],
				[30, 20, 1, 0]]
sample_decisions_b = gaussian_minimum_expected_loss_decisions(samples_q2, loss_matrix_b, means, covs, priors)

# Create and print confusion matrix
confusion_matrix_b = generate_confusion_matrix(sample_decisions_b, sample_labels_q2, len(means))
print("Part B Confusion Matrix (Predicted Class on Rows)\n" + str(confusion_matrix_b))

# Create 2D scatter plot for Part B
scatter_plot(samples_q2, sample_labels_q2, sample_decisions_b, len(means),
			 "Given Loss X-Y Scatterplot with Classes and Predictions")

# Calculate minimum expected loss from confusion matrix counts and loss matrix
confusion_matrix_counts_b = generate_confusion_matrix_counts(sample_decisions_b, sample_labels_q2, len(means))
minimum_expected_loss = sum(multiply(confusion_matrix_counts_b, loss_matrix_b)) / NUM_GENERATED_SAMPLES
print("Minimum Expected Loss: " + str(minimum_expected_loss))