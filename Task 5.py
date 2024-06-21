#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the dataset
car_data = pd.read_csv('CarPrice_Assignment.csv')

# Display the first few rows of the dataset
print(car_data.head())

# Display the dataset summary
print(car_data.info())
print(car_data.describe())


# In[6]:


# Check for missing values
print(car_data.isnull().sum())

# Identify numerical columns
numerical_cols = car_data.select_dtypes(include=['float', 'int']).columns

# Fill missing values in numerical columns with their mean
car_data[numerical_cols] = car_data[numerical_cols].fillna(car_data[numerical_cols].mean())

# Convert categorical columns to numerical using one-hot encoding
car_data = pd.get_dummies(car_data, drop_first=True)

# Display the first few rows of the preprocessed dataset
print(car_data.head())


# In[7]:


from sklearn.model_selection import train_test_split

# Define the feature set and the target variable
X = car_data.drop('price', axis=1)  # assuming 'price' is the target column
Y = car_data['price']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[8]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, Y_train)

# Predict on the test set
Y_pred = model.predict(X_test)


# In[9]:


# Evaluate the model
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = mse ** 0.5
r2 = r2_score(Y_test, Y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R^2: {r2}')


# In[18]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Initialize the model
rf_model = RandomForestRegressor(random_state=42)

# Define hyperparameters to tune
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30]
}

# Perform Grid Search
grid_search = GridSearchCV(rf_model, param_grid=params, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, Y_train)

# Best parameters
print("Best parameters:", grid_search.best_params_)

# Train the best model
best_rf_model = grid_search.best_estimator_

# Predict on the test set
Y_pred_rf = best_rf_model.predict(X_test)




# In[ ]:


# Evaluate the model
mae_rf = mean_absolute_error(Y_test, Y_pred_rf)
mse_rf = mean_squared_error(Y_test, Y_pred_rf)
rmse_rf = mse_rf ** 0.5
r2_rf = r2_score(Y_test, Y_pred_rf)

print(f'Random Forest MAE: {mae_rf}')
print(f'Random Forest MSE: {mse_rf}')
print(f'Random Forest RMSE: {rmse_rf}')
print(f'Random Forest R^2: {r2_rf}')


# In[11]:


from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = best_rf_model.predict([list(data.values())])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




