import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

st.title("Social Impact Sentiment Analyzer")

# Text input area for multiple feedbacks separated by new lines
feedback_text = st.text_area("Enter feedback texts (one per line)")

if feedback_text:
    # Split text input into list of feedback
    feedback_list = feedback_text.strip().split('\n')
    df = pd.DataFrame({'feedback': feedback_list})

    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    # Analyze each feedback
    df['sentiment'] = df['feedback'].apply(lambda x: get_sentiment(analyzer.polarity_scores(x)['compound']))

    st.write("### Sentiment Classification")
    st.dataframe(df)

    # Count sentiments
    sentiment_counts = df['sentiment'].value_counts()

    # Plot bar chart
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='bar', ax=ax, color=['green', 'gray', 'red'])
    plt.title('Social Impact Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    st.pyplot(fig)
