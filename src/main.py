"""Decision Tree problem in Diabetic patients data

Alireza Soltani Neshan
Thu Azar 10 
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Reading `diabetes` csv data
diabetes = pd.read_csv("../data/diabetes.csv")

# Checks all value of features if was `None`
print(diabetes.isna().sum())

# Check verbose of data
print(diabetes.info())

indian_diabetes = pd.read_csv(
    "../data/pima-indians-diabetes.csv", error_bad_lines=False
)

print(indian_diabetes)

# Check value of `Outcome` feature (767 data and check how many Outcome value is
# 0 and how many is 1)
print(diabetes["Outcome"].value_counts())

# 0 -> 500, 1 -> 268
x_a = diabetes.drop("Outcome", axis=1)
y_a = diabetes["Outcome"]

# Read first 5 rows
print(x_a.head())
print(y_a.head())

# Pareto for data split between 80% and 20%
x_train, x_test, y_train, y_test = train_test_split(x_a, y_a, test_size=0.2)

print(f"Train on X_a -> {x_train.shape}")
print(f"Train on Y_a -> {y_train.shape}")
print(f"Test of X -> {x_test.shape}")
print(f"Test of Y -> {y_test.shape}")

decision_tree_classifier = DecisionTreeClassifier()

# Fit and make model by this data
decision_tree_classifier.fit(x_train, y_train)

# Testing the model
print(decision_tree_classifier.score(x_test, y_test))
