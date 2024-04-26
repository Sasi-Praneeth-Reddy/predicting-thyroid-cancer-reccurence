import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc

# Load the dataset
data = pd.read_csv("Thyroid_Diff.csv")

# Encode categorical variables using one-hot encoding
categorical_features = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Thyroid Function', 
                        'Physical Examination', 'Adenopathy', 'Pathology', 'Focality', 'Risk', 
                        'T', 'N', 'M', 'Stage', 'Response']
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Encode the 'Recurred' variable into numerical labels
label_encoder = LabelEncoder()
data['Recurred'] = label_encoder.fit_transform(data['Recurred'])

# Splitting the data into features (X) and target variable (y)
X = data.drop(columns=['Recurred'])  # Features
y = data['Recurred']  # Target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Size of the training set
train_size = X_train.shape[0]

# Size of the test set
test_size = X_test.shape[0]

print("Training set size:", train_size)
print("Test set size:", test_size)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Binarize features
X_train_bin = (X_train_scaled > 0).astype(int)
X_test_bin = (X_test_scaled > 0).astype(int)

# Custom Bernoulli Naive Bayes Classifier
class bernoullinb:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        n_classes = np.max(y) + 1
        n_features = X.shape[1]
        self.class_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for c in range(n_classes):
            class_indices = np.where(y == c)[0]
            class_count = len(class_indices)
            self.class_prior_[c] = (class_count + self.alpha) / (len(y) + n_classes * self.alpha)

            for f in range(n_features):
                feature_count = np.sum(X[class_indices, f])
                self.feature_log_prob_[c, f] = np.log((feature_count + self.alpha) / (class_count + 2 * self.alpha))

    def _joint_log_likelihood(self, X):
        return np.dot(X, self.feature_log_prob_.T) + np.log(self.class_prior_)

    def predict(self, X):
        joint_log_likelihood = self._joint_log_likelihood(X)
        return np.argmax(joint_log_likelihood, axis=1)

# Training Custom Bernoulli Naive Bayes classifier
nb_classifier = bernoullinb()
nb_classifier.fit(X_train_bin, y_train)

# Evaluate Custom Bernoulli Naive Bayes classifier
y_pred_nb = nb_classifier.predict(X_test_bin)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Bernoulli Naive Bayes Classifier Accuracy:", accuracy_nb)

# Custom K-Nearest Neighbors Classifier
class KNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            nearest_neighbors = np.argsort(distances)[:self.n_neighbors]
            y_nearest = self.y_train.iloc[nearest_neighbors]  # Use iloc to index DataFrame by integer location
            unique, counts = np.unique(y_nearest, return_counts=True)
            y_pred.append(unique[np.argmax(counts)])
        return y_pred


# Training Custom KNN classifier
knn_classifier = KNN(n_neighbors=5)  # You may adjust the number of neighbors
knn_classifier.fit(X_train_scaled, y_train)

# Evaluate Custom KNN classifier
y_pred_knn = knn_classifier.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Classifier Accuracy:", accuracy_knn)


# Training Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluate the trained Random Forest classifier
y_pred_rf = rf_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Classifier Accuracy:", accuracy_rf)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

# Confusion matrix for Custom Bernoulli Naive Bayes classifier
plot_confusion_matrix(y_test, y_pred_nb, title="Bernoulli Naive Bayes Confusion Matrix")

# Confusion matrix for Custom KNN classifier
plot_confusion_matrix(y_test, y_pred_knn, title="KNN Confusion Matrix")

# Confusion matrix for Random Forest classifier
plot_confusion_matrix(y_test, y_pred_rf, title="Random Forest Confusion Matrix")

# Define the filename for the output text file
output_filename = "predictions.txt"

# Open the file in write mode
with open(output_filename, 'w') as file:
    # Writing predictions for Custom Bernoulli Naive Bayes classifier
    file.write("Bernoulli Naive Bayes Classifier Predictions:\n")
    for pred in y_pred_nb:
        file.write(str(pred) + '\n')
    file.write('\n')
    
    # Writing predictions for Custom KNN classifier
    file.write("KNN Classifier Predictions:\n")
    for pred in y_pred_knn:
        file.write(str(pred) + '\n')
    file.write('\n')
    
    # Writing predictions for Random Forest classifier
    file.write("Random Forest Classifier Predictions:\n")
    for pred in y_pred_rf:
        file.write(str(pred) + '\n')

# Print message indicating the file write completion
print("Predictions have been written to", output_filename)

# Classification report for Custom Bernoulli Naive Bayes classifier
print("Bernoulli Naive Bayes Classifier Report:")
print(classification_report(y_test, y_pred_nb))

# Area under the ROC curve for Custom Bernoulli Naive Bayes classifier
auc_nb = roc_auc_score(y_test, y_pred_nb)
print("Bernoulli Naive Bayes Classifier AUC:", auc_nb)

# Classification report for Custom KNN classifier
print("KNN Classifier Report:")
print(classification_report(y_test, y_pred_knn))

# Area under the ROC curve for Custom KNN classifier
auc_knn = roc_auc_score(y_test, y_pred_knn)
print("KNN Classifier AUC:", auc_knn)

# Classification report for Random Forest classifier
print("Random Forest Classifier Report:")
print(classification_report(y_test, y_pred_rf))

# Area under the ROC curve for Random Forest classifier
auc_rf = roc_auc_score(y_test, y_pred_rf)
print("Random Forest Classifier AUC:", auc_rf)

# Ploting ROC curve for Custom Bernoulli Naive Bayes classifier
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_pred_nb)
roc_auc_nb = auc(fpr_nb, tpr_nb)
plt.figure(figsize=(8, 6))
plt.plot(fpr_nb, tpr_nb, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_nb)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Bernoulli Naive Bayes Classifier')
plt.legend(loc="lower right")
plt.show()

# Ploting ROC curve for Custom KNN classifier
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)
plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_knn)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - KNN Classifier')
plt.legend(loc="lower right")
plt.show()

# Ploting ROC curve for Random Forest classifier
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest Classifier')
plt.legend(loc="lower right")
plt.show()

# Bar graph for accuracy comparison
classifiers = ['Bernoulli Naive Bayes', 'KNN', 'Random Forest']
accuracies = [accuracy_nb, accuracy_knn, accuracy_rf]

plt.figure(figsize=(10, 6))
plt.bar(classifiers, accuracies, color='skyblue')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Classifiers')
plt.ylim(0, 1)  # Set y-axis limit to 0-1 for accuracy
plt.show()

# Bar graph for AUC comparison
auc_scores = [auc_nb, auc_knn, auc_rf]

plt.figure(figsize=(10, 6))
plt.bar(classifiers, auc_scores, color='lightgreen')
plt.xlabel('Classifier')
plt.ylabel('AUC Score')
plt.title('AUC Score Comparison of Classifiers')
plt.ylim(0, 1)  # Set y-axis limit to 0-1 for AUC
plt.show()
