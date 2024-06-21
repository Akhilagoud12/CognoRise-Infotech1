#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
data= pd.read_csv('TeePublic_review.csv', encoding='latin-1', on_bad_lines='skip') 

# Basic Data Exploration
data.head()


# In[20]:


get_ipython().system('pip install TextBlob')
from textblob import TextBlob

# Function to get sentiment polarity
def get_sentiment(review):
    # Handle potential non-string values
    if isinstance(review, str):
        analysis = TextBlob(review)
        return analysis.sentiment.polarity
    else:
        return None  # Or any other default value you prefer

# Apply sentiment function
data['sentiment'] = data['review'].apply(get_sentiment)

# Categorize sentiment (handling potential None values)
data['sentiment_category'] = data['sentiment'].apply(lambda x: 'positive' if x is not None and x > 0 else ('negative' if x is not None and x < 0 else 'neutral'))

# Display sentiment distribution
print(data['sentiment_category'].value_counts())


# In[ ]:





# In[ ]:




