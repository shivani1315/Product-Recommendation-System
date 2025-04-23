import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image

# Download tokenizer
nltk.download('punkt')

# Load dataset
df = pd.read_csv("amazon_product.csv")
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Initialize stemmer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')

# Tokenization and stemming function
def tokenize_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha()]
    stemmed = [stemmer.stem(w) for w in tokens]
    return " ".join(stemmed)

# Preprocess dataset
df['stemmed_token'] = df.apply(lambda row: tokenize_stem(str(row['Title']) + " " + str(row['Description'])), axis=1)

# TF-IDF
tfidfv = TfidfVectorizer()
tfidf_matrix = tfidfv.fit_transform(df['stemmed_token'])

# Search function
def search_product(query):
    stemmed_query = tokenize_stem(query)
    query_vec = tfidfv.transform([stemmed_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    df['similarity'] = similarities
    results = df.sort_values(by='similarity', ascending=False).head(10)
    return results[['Title', 'Description', 'Category']]

# Streamlit UI Styling
st.markdown("""
    <style>
    .title {
        font-size: 42px;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Optional banner image
img = Image.open('Image.png')  # Make sure the image exists
st.image(img, use_column_width=True)

# Title and input
st.markdown('<div class="title">🔍 Product Search & Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Type a product name or description to find similar items</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    query = st.text_input("Enter product name", placeholder="e.g. wireless earbuds with long battery life")
    submit = st.button("Search")

# Output results as table
if submit and query.strip() != "":
    results = search_product(query)
    st.markdown("### 🔎 Top 10 Matching Products:")
    st.dataframe(results.reset_index(drop=True), use_container_width=True)
