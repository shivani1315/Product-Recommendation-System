# 🔍 Product Search & Recommendation System

Product Search &amp; Recommendation System is a Streamlit-based web application that uses Natural Language Processing (NLP) and machine learning to help users search for products and receive content-based recommendations. 

A content-based recommendation system built using Streamlit, TF-IDF, and NLTK. Users can search for product titles or descriptions, and the app returns the most relevant products.

## 🚀 Features

- Search for similar products using text input
- Cosine similarity over TF-IDF vectors
- NLP preprocessing (tokenizing, stemming)
- Tabular output of top matching products
- Product dataset with categories

## 📦 Dataset

Make sure `amazon_product.csv` is present in the same directory. Columns used:
- `Title`
- `Description`
- `Category`
- `Image` (optional for later image support)
