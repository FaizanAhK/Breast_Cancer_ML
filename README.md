Each year number of deaths is increasing extremely because of breast cancer. It is the most frequent type of all cancers and the major cause of death in women worldwide. Any development for prediction and diagnosis of cancer disease is capital important for a healthy life. Consequently, high accuracy in cancer prediction is important to update the treatment aspect and the survivability standard of patients.

Machine learning techniques can bring a large contribute on the process of prediction and early diagnosis of breast cancer, became a research hotspot and has been proved as a strong technique. 

In this study, we applied five machine learning algorithms: Support Vector Machine (SVM), Random Forest, Logistic Regression, Decision tree (C4.5) and K-Nearest Neighbours (KNN) on the Breast Cancer Wisconsin Diagnostic dataset, after obtaining the results, a performance evaluation and comparison is carried out between these different classifiers. The main objective of this research paper is to predict and diagnosis breast cancer, using machine-learning algorithms, and find out the most effective whit respect to confusion matrix, accuracy and precision. 

It is observed that Support vector Machine outperformed all other classifiers and achieved the highest accuracy (97.2%).All the work is done in the Anaconda environment based on python programming language and Scikit-learn library

# OVERVIEW

Objective:

To predict whether a tumor is malignant or benign using the Breast Cancer Wisconsin dataset, employing machine learning models for classification.

Dataset:

Name: Breast Cancer Wisconsin Dataset
Description: This dataset contains features computed from a digitized image of a breast mass, describing the characteristics of cell nuclei present.

Features:

Independent Variables: 30 numerical features such as radius, texture, smoothness, compactness, etc.
Target Variable: Binary classification - Malignant (1) or Benign (0).

Steps in the Project:
1. Data Preprocessing:
Handling Missing Values: Checked and handled any null values (if present) in the dataset.
Normalization/Scaling: Applied feature scaling (e.g., MinMaxScaler, StandardScaler) to ensure all features are on a similar scale for optimal model performance.
Feature Selection: Used techniques like correlation heatmaps or feature importance from models to identify significant features.

2. Exploratory Data Analysis (EDA):
Visualized feature distributions to understand data patterns (e.g., histograms, pairplots).
Analyzed relationships between features and target variables.
Used heatmaps to study correlations between features to avoid multicollinearity.

3. Model Implementation:
Implemented the following machine learning models:
Logistic Regression: For a baseline understanding of the classification problem.
Support Vector Machine (SVM): Leveraged its strength in separating classes with a clear margin.
Random Forest Classifier: Used to analyze feature importance and handle non-linear relationships.
K-Nearest Neighbors (KNN): A simple model to compare results.

4. Model Evaluation:
Evaluated model performance using:
Confusion Matrix: For understanding true positives, true negatives, false positives, and false negatives.
Accuracy: Percentage of correct predictions.
Precision, Recall, F1-Score: To measure the model's reliability and balance between sensitivity and specificity.
ROC-AUC Curve: To visualize the trade-off between sensitivity and specificity.

5. Results:
Compared the performance of all models based on accuracy and other metrics.
Identified the best-performing model for predicting breast cancer based on the dataset.

6. Visualization:
Visualized important features using bar charts (e.g., from Random Forest feature importances).
Plotted decision boundaries for models like SVM or KNN (if applicable).
Included ROC-AUC curves for the top-performing models.

