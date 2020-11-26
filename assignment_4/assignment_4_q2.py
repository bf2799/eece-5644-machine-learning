import numpy as np
import sklearn as skl
from sklearn import model_selection
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def read_data_from_csv(file_name):
    """
    Reads data from CSV in format (x_samples, y_samples)
    Outputs 2 numpy arrays for sample inputs and target outputs
    :param file_name: Name of file to read from
    :return: Numpy array of inputs, shape (num_samples, num_features) and array of outputs, shape (num_samples,)
    """
    file = open(file_name, 'r').readlines()
    file = np.array([line[:-1].split(',') for line in file])
    new_file = []
    for line in range(len(file)):
        new_file.append([float(sample) for sample in file[line]])
    input_array = np.transpose(new_file[:-1])
    return input_array, np.transpose(new_file[-1])


def plot_2d_class_scatter(data, labels, title):
    class_colors = ['black' if label == 1 else 'red' for label in labels]
    plt.scatter([row[0] for row in data], [row[1] for row in data], s=1, c=class_colors)
    plt.title(title)
    plt.xlabel('X-1')
    plt.ylabel('X-2')
    legend_elements = [Line2D([0], [0], marker='o', color='white', label='Class 1 Samples', markerfacecolor='black',
                              markersize=5),
                       Line2D([0], [0], marker='o', color='white', label='Class 2 Samples', markerfacecolor='red',
                              markersize=5)]
    plt.legend(handles=legend_elements, loc='lower left')
    plt.show()


# Get training and test data from CSVs
train_data, train_labels = read_data_from_csv('train_data_q2.csv')
test_data, test_labels = read_data_from_csv('test_data_q2.csv')

# Plot training data and test data with classes
plot_2d_class_scatter(train_data, train_labels, 'Distribution of Generated Training Samples')
plot_2d_class_scatter(test_data, test_labels, 'Distribution of Generated Test Samples')

# Get test values for C and sigma as 0.001 to 1000 on log scale
c_test_vals = np.logspace(start=-3, stop=3, num=7, base=10.0)
sigma_test_vals = c_test_vals.copy()

# Perform 10-fold cross validation to find best values for hyperparameters C and sigma for SVC
# Then, re-fit best model as parameter to GridSearchCV
train_split = skl.model_selection.KFold(n_splits=10)
k_fold_param_dict = {'C': c_test_vals, 'kernel': ['rbf'], 'gamma': sigma_test_vals, 'tol': [1e-4]}
svm_model = skl.model_selection.GridSearchCV(estimator=skl.svm.SVC(), param_grid=k_fold_param_dict,
                                             scoring='accuracy', cv=train_split, refit=True, verbose=1)
svm_model.fit(X=train_data, y=train_labels)

# Print best C & sigma
cv_results = svm_model.cv_results_
print('Best Estimator: ' + str(cv_results['params'][svm_model.best_index_]))
# Print probability of correct classification for training data
print('Train Probability of Correct Classification: ' + str(svm_model.best_score_))

# Graph probability of correct classification for each C, sigma
accuracy_vals = cv_results.get('mean_test_score')
for i in range(c_test_vals.shape[0]):
    temp_accuracy_vals = accuracy_vals[i * sigma_test_vals.shape[0]:(i + 1) * sigma_test_vals.shape[0]]
    plt.plot(sigma_test_vals, temp_accuracy_vals)
plt.xscale('log')
plt.xlabel('Sigma')
plt.ylabel('Probability Correct Classification')
plt.title('Probability of Correct Classification for Various C and Sigma Values')
legend_entries = ['C = ' + str(c) for c in c_test_vals]
plt.legend(legend_entries, loc='upper right')
plt.show()

# Print probability of correct classification on test data
print('Test Probability of Correct Classification: ' + str(svm_model.score(X=test_data, y=test_labels)))

# Plot correctly and incorrectly classified data points in space
test_prediction_labels = svm_model.predict(test_data)
class_colors = []
for label in range(test_labels.shape[0]):
    if test_prediction_labels[label] == test_labels[label]:
        if test_labels[label] == 1:
            class_colors.append('green')
        else:
            class_colors.append('limegreen')
    else:
        if test_labels[label] == 1:
            class_colors.append('darkred')
        else:
            class_colors.append('red')
plt.scatter([row[0] for row in test_data], [row[1] for row in test_data], s=1, c=class_colors)
plt.title('Classification Correctness of Test Samples')
plt.xlabel('X-1')
plt.ylabel('X-2')
legend_elements = [Line2D([0], [0], marker='o', color='white', label='Correct, Class 1', markerfacecolor='green',
                          markersize=5),
                   Line2D([0], [0], marker='o', color='white', label='Correct, Class 2', markerfacecolor='limegreen',
                          markersize=5),
                   Line2D([0], [0], marker='o', color='white', label='Incorrect, Class 1', markerfacecolor='darkred',
                          markersize=5),
                   Line2D([0], [0], marker='o', color='white', label='Incorrect, Class 2', markerfacecolor='red',
                          markersize=5)]
plt.legend(handles=legend_elements, loc='lower left')
plt.show()

