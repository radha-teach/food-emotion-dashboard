import streamlit as st
import pandas as pd
from textblob import TextBlob
from transformers import pipeline

# Load data
df = pd.read_csv("reviews.csv")

# Sentiment
def get_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

# Emotion model
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

def get_emotion(text):
    return emotion_model(text)[0]['label']

df["sentiment"] = df["review"].apply(get_sentiment)

st.title("🍜 Food Emotion Dashboard")

# Filter
cuisine = st.selectbox("Select Cuisine", df["cuisine"].unique())
filtered_df = df[df["cuisine"] == cuisine]

st.write(filtered_df)

# Chart
st.subheader("Sentiment Count")
st.bar_chart(filtered_df["sentiment"].value_counts())

# Analyzer
st.subheader("Analyze Your Review")
user_input = st.text_area("Enter review")

if st.button("Analyze"):
    st.write("Sentiment:", get_sentiment(user_input))
    st.write("Emotion:", get_emotion(user_input))
