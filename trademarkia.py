#!/usr/bin/env python
# coding: utf-8

# In[3]:


import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open('idmanual.json', 'r') as f:
    data = json.load(f)
train_data, test_data = train_test_split(data, test_size=0.2)

vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform([item['description'] for item in train_data])

def recommend_classes(input_description):
    input_features = vectorizer.transform([input_description])
    similarity_scores = cosine_similarity(input_features, train_features)
    # Find the most similar classes based on the similarity scores
    recommended_classes = train_data[similarity_scores.argmax()]['class_id']
    return recommended_classes


input_description = "Description of the user's goods or services"
recommended_classes = recommend_classes(input_description)
print("Recommended classes:", recommended_classes)


# In[ ]:




