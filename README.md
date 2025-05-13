# 9.Exercise-session-Nonparametric-estimation
Kernel Density-Based Statistical Classification
📊 Kernel Density-Based Statistical Classification
This project focuses on non-parametric classification using Gaussian kernel density estimation. It guides the implementation of a kdclassify function in Matlab or Python to classify samples based on empirical probability distributions of each class. With flexible input parameters including training data, class labels, test data, and bandwidth control (h), the method estimates class-wise likelihoods and assigns labels based on maximum posterior probability. Ideal for learners in machine learning and statistical signal processing, this project emphasizes a data-driven approach to classification without assuming fixed model parameters, making it well-suited for handling complex or non-linear class boundaries.
![image](https://github.com/user-attachments/assets/ea73440c-474b-4ce4-8ad2-336f43a13b48)
Construct a Matlab/Python function for statistical classification using empirical probability density functions based on Gaussian kernel density estimates. The function call should be:

matlab
Copy
Edit
C = kdclassify(traindata, trainclass, data, h)
and the parameters and the output should be as follows:

Matrix traindata contains training examples so that each column is a single example.

Row vector trainclass contains the classes of the examples, so that element i of trainclass is the class of the example in column i of traindata.

Matrix data contains samples to be classified, one in each column.

h is the length parameter which determines the effective width of the Gaussian.

Row vector C contains the classes and it should include one value for each column in data.

Verify experimentally that the implementation works using the provided data: CSV, MAT.

Hints: The algorithm works as follows:

For each sample to be classified, determine the probability of the sample to belong to each class as follows (the sum is over samples in a class):

where x is a sample to be classified, h is the length parameter, l is the dimensionality, N is the number of samples in the class-specific training set, φ is the indicator function and xᵢ is a training set sample from a specific class. For the indicator function φ, standardized normal density (zero mean, unit variance) should be used.

Choose the class based on the maximum probability.

```python
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

################ part1: Load and preview the data file ################
data_path = 'C:/12.LUT/00.Termic Cources/2.pattern recognition/2.part2/1.week8 - mixtures/2.code/t024.csv'
data_frame = pd.read_csv(data_path, header=None)
data_frame.head()
################ part2: Prepare arrays for features and classes ################
data_array = data_frame.values
################ part3: Separate feature data and class labels ################
feature_data = data_array[:, :-1].T 
class_labels = data_array[:, -1] 
test_data = feature_data
################ part4: Define the Gaussian kernel function ################
def gaussian_density(sample, reference_sample, bandwidth):
    # Compute the Euclidean distance between points and normalize
    distance = np.linalg.norm(sample - reference_sample) / bandwidth
    return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * distance**2)
################ part5: Define the custom classifier function ################
def custom_classifier(features, labels, samples, bandwidth):
    predictions = []
    for sample in samples.T: 
        class_probabilities = {}
        for unique_class in np.unique(labels):
            class_data = features[:, labels == unique_class] 
            num_samples = class_data.shape[1] 
            probability = np.mean([gaussian_density(sample, reference_sample, bandwidth) 
                                   for reference_sample in class_data.T])
            class_probabilities[unique_class] = probability
        predictions.append(max(class_probabilities, key=class_probabilities.get))
    return np.array(predictions)

################ part6: Run classifier and display results ################
bandwidth_value = 2.0
predicted_labels = custom_classifier(feature_data, class_labels, test_data, bandwidth_value)
print("Predicted classes by Arman Golbidi's classifier:", predicted_labels)

################ part7: Plot the results in a vertical layout ################
print("Feature data shape:", feature_data.shape)
print("Class labels shape:", class_labels.shape)
if feature_data.shape[0] > feature_data.shape[1]:
    feature_data = feature_data.T
plt.figure(figsize=(10, 14))
plt.subplot(2, 1, 1)
for unique_class in np.unique(class_labels):
    class_indices = class_labels == unique_class
    plt.scatter(feature_data[0, class_indices], feature_data[1, class_indices], 
                label=f"Actual Class {int(unique_class)}", marker='s', color='darkcyan' if unique_class == 1 else 'purple')
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Actual Classes in Training Data")
plt.legend(title="Legend", loc="upper right", fontsize='small', title_fontsize='medium')
plt.subplot(2, 1, 2)
for unique_class in np.unique(predicted_labels):
    class_indices = predicted_labels == unique_class
    plt.scatter(test_data[0, class_indices], test_data[1, class_indices], 
                label=f"Predicted Class {int(unique_class)}", marker='^', color='coral' if unique_class == 1 else 'darkgreen')
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Predicted Classes for Test Data")
plt.legend(title="Legend", loc="upper right", fontsize='small', title_fontsize='medium')
plt.tight_layout()
plt.show()
```
