#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load the Iris dataset
iris = pd.read_csv('IRIS.csv')

# Features (sepal and petal measurements)
X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

# Target (species)
y = iris['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
iris


# In[5]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

accuracy, report, conf_matrix


# In[3]:


# Example of new data (sepal length, sepal width, petal length, petal width)
new_data = [[5.1, 3.5, 1.4, 0.2]]

# Predict the species
predicted_species = model.predict(new_data)
predicted_species


# In[ ]:




