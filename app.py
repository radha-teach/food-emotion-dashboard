import streamlit as st
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')

from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Food Emotion Intelligence", layout="wide")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/yelp_cleaned_sample.csv")

df = load_data()

# ---------------- SENTIMENT ----------------
def get_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["review"].apply(get_sentiment)

# ---------------- HEADER ----------------
st.title("🍜 Food Emotion Intelligence Dashboard")
st.markdown("### 🌍 Explore how people feel about food across the world")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Filters")
selected_cuisine = st.sidebar.selectbox("Select Cuisine", df["cuisine"].unique())

filtered_df = df[df["cuisine"] == selected_cuisine]

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Reviews", len(filtered_df))
col2.metric("Avg Rating", round(filtered_df["rating"].mean(), 2))
col3.metric("Positive %", 
            round((filtered_df["sentiment"] == "Positive").mean()*100, 1))

# ---------------- WORLD MAP ----------------
st.subheader("🌍 Global Food Sentiment Map")

# Fake coordinates mapping (simple demo)
location_coords = {
    "India": [20.5937, 78.9629],
    "USA": [37.0902, -95.7129],
    "Japan": [36.2048, 138.2529],
    "Italy": [41.8719, 12.5674],
    "Mexico": [23.6345, -102.5528]
}

df["lat"] = df["location"].map(lambda x: location_coords.get(x, [0,0])[0])
df["lon"] = df["location"].map(lambda x: location_coords.get(x, [0,0])[1])

st.map(df[["lat", "lon"]])

# ---------------- SENTIMENT CHART ----------------
st.subheader("📊 Sentiment Distribution")
st.bar_chart(filtered_df["sentiment"].value_counts())

# ---------------- ML MODEL (TF-IDF SIMILARITY) ----------------
st.subheader("🤖 Food Recommendation System")

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df["review"])

def recommend_food(user_input):
    user_vec = tfidf.transform([user_input])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    idx = similarity.argsort()[0][-3:][::-1]
    return df.iloc[idx][["review", "cuisine"]]

user_input = st.text_input("Enter what kind of food experience you want:")

if user_input:
    recs = recommend_food(user_input)
    st.write("### 🔥 Recommended Reviews:")
    st.write(recs)

# ---------------- ADVANCED ANALYSIS ----------------
st.subheader("📈 Rating vs Sentiment")

chart_data = filtered_df.groupby("sentiment")["rating"].mean()
st.bar_chart(chart_data)

# ---------------- REVIEW ANALYZER ----------------
st.subheader("📝 Analyze Your Review")

review = st.text_area("Type your review here:")

if st.button("Analyze"):
    sentiment = get_sentiment(review)
    st.success(f"Sentiment: {sentiment}")

# ---------------- UI FOOTER ----------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | NLP | ML")
