import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Read all CSV files in the dataNASA folder
folder_path = 'D:\ISE\Challenge Task 1- Software Defect Prediction\data\Software Defect Prediction - Data\dataNASA'
all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

# Concatenate all dataframes
data = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
#data = pd.read_csv('D:\ISE\Challenge Task 1- Software Defect Prediction\data\Software Defect Prediction - Data\dataNASA\KC3.csv')
# Drop the 'id' column
data = data.drop(columns=['id'])


# Show basic information
print(data.info())
print(data.head())

# Map 'Defective' values 'Y' to 1 and 'N' to 0
data['Defective'] = data['Defective'].map({'Y': 1, 'N': 0})

# Verify the mapping
print(data['Defective'].value_counts())

# Plot the distribution of the target variable
sns.countplot(x='Defective', data=data)  # Replace 'target_column' with the actual column name
plt.title('Distribution of Defective')
plt.show()

# Correlation matrix
plt.figure(figsize=(20,20))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=2)
plt.title('Correlation Heatmap')
plt.show()

# Calculate the correlation of all features with the 'Defective' column
correlation_with_defective = correlation_matrix['Defective'].abs().sort_values(ascending=False)

# Get the top 10 features with the highest correlation to 'Defective'
top_10_features = correlation_with_defective.index[1:11]  # Exclude 'Defective' itself
print("Top 10 features with highest correlation to 'Defective':")
print(top_10_features)

# Draw bar plot to present correlation value of each feature to Defective
plt.figure(figsize=(12, 8))
sns.barplot(x=correlation_with_defective[top_10_features].values, y=top_10_features)
plt.xlabel('Correlation with Defective')
plt.ylabel('Features')
plt.title('Top 10 Features Correlated with Defective')
plt.show()

# Draw heatmap of features to see their values
plt.figure(figsize=(16, 9))
sns.heatmap(data)
plt.title('Heatmap of Feature')
plt.show()

X = data.drop('Defective', axis=1)
X = (X-X.min())/(X.max()-X.min())
X
X.describe()

plt.figure(figsize=(20, 10))
sns.boxplot(data=X)
plt.xticks(rotation=90)
plt.title('Box Plot of Features in X')
plt.show()

y = data['Defective']
y

# Check for missing values
#print(X.isnull().sum())

# Option 1: Fill missing values with the mean of each column
X = X.fillna(X.mean())

# Option 2: Drop rows with missing values
#X = X.dropna()
#y = y[X.index]
#print(y.count())
# Option 3: Forward fill missing values
# data = data.fillna(method='ffill')

plt.figure(figsize=(16, 9))
sns.heatmap(X)