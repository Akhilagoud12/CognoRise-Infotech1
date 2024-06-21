#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
titanic = pd.read_csv('tested.csv')  
data.head()


# In[11]:


features = ['Pclass', 'Sex', 'Age', 'Fare']
X = titanic[features]
X['Age'].fillna(X['Age'].mean(), inplace=True)
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X.fillna(X.mean(), inplace=True)
y = titanic['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
plt.figure(figsize=(8, 6))
titanic.groupby('Pclass')['Survived'].mean().plot(kind='bar')
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.show()


# In[ ]:





# In[ ]:




