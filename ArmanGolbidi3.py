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
