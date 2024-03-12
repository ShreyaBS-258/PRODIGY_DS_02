# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the train, test, and gender submission datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
gender_submission_data = pd.read_csv('gender_submission.csv')

# Display the first few rows of each dataset
print("Train Data:")
print(train_data.head())

print("\nTest Data:")
print(test_data.head())

print("\nGender Submission Data:")
print(gender_submission_data.head())

# Check for missing values in each dataset
print("\nMissing Values in Train Data:")
print(train_data.isnull().sum())

print("\nMissing Values in Test Data:")
print(test_data.isnull().sum())

print("\nMissing Values in Gender Submission Data:")
print(gender_submission_data.isnull().sum())

# Data Cleaning
# Handle missing values
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

# Remove unnecessary columns
train_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Data Formatting
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data['Embarked'] = test_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Exploratory Data Analysis (EDA)
# Summary Statistics
print("\nSummary Statistics for Train Data:")
print(train_data.describe())

# Univariate Analysis
plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', data=train_data)
plt.title('Survival Count in Train Data')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(train_data['Age'], bins=20, kde=True)
plt.title('Age Distribution in Train Data')
plt.show()

# Bivariate Analysis
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Fare', data=train_data)
plt.title('Fare by Passenger Class in Train Data')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=train_data)
plt.title('Survival by Age and Fare in Train Data')
plt.show()

# Multivariate Analysis
plt.figure(figsize=(10, 6))
sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for Train Data')
plt.show()
