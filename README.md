# Software-Defect-Prediction

## Description:
This project focuses on analyzing and classifying NASA-related data using various machine learning techniques. The dataset undergoes preprocessing, including handling missing values, feature selection, and data balancing, to improve model performance. The goal is to develop a robust classification model that can accurately predict outcomes based on historical data.

## Methods:

### Data Preprocessing:

Data loading from multiple CSV files.

Handling missing values using KNN imputation.

Standardization of features using StandardScaler.

Feature selection using SelectKBest with ANOVA F-test.

Data balancing using SMOTE to handle class imbalances.

### Model Training:

Implemented classifiers: Random Forest, Support Vector Machine (SVM), Logistic Regression, and Naive Bayes.

Combined models using a Voting Classifier for enhanced performance.

Hyperparameter tuning and cross-validation using StratifiedKFold.

### Model Evaluation:

Performance metrics: Accuracy, F1-score, ROC-AUC, and Geometric Mean Score.

Comparison of different classifiers to identify the best-performing model.

## Achievements:

Successfully processed and cleaned large datasets from NASA sources.

Improved classification performance through feature selection and data balancing techniques.

Developed an ensemble model that outperformed individual classifiers in predictive accuracy.

Achieved high classification scores, demonstrating the model’s reliability in NASA data prediction tasks.

Provided insights into the most influential features for classification, aiding in further research and development.

