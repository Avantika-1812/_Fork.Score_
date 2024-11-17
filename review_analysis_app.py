import streamlit as st
from transformers import RobertaForSequenceClassification, RobertaTokenizer, pipeline

# Load your trained RoBERTa model
roberta_model = RobertaForSequenceClassification.from_pretrained(r"my_models\roberta")
roberta_tokenizer = RobertaTokenizer.from_pretrained(r"my_models\roberta")
roberta_pipeline = pipeline("sentiment-analysis", model=roberta_model, tokenizer=roberta_tokenizer)

# Mapping from sentiment labels to primary sentiment and emotions with emojis
label_emotion_mapping = {
    'LABEL_0': {'primary': 'Negative', 'emotion': 'Sadness', 'emoji': '😞'},
    'LABEL_1': {'primary': 'Neutral', 'emotion': 'Calmness', 'emoji': '😐'},
    'LABEL_2': {'primary': 'Positive', 'emotion': 'Joy', 'emoji': '😊'}
}

# CSS for animated gradient background in Streamlit container
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(45deg, #ff6ec4, #7873f5);
        background-size: 200% 200%;
        animation: gradient 10s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main app function
def main_app():
    st.title("Review Analysis")

    review_text = st.text_area("Enter the review text for analysis", height=200)

    if st.button("Analyze"):
        if review_text:
            result = roberta_pipeline(review_text)
            label = result[0]['label']
            confidence = result[0]['score']

            # Get primary emotion, associated sentiment, and emoji
            emotions = label_emotion_mapping[label]
            primary_emotion = emotions['primary']
            emotion = emotions['emotion']
            emoji = emotions['emoji']

            # Display primary sentiment with emoji, confidence, and associated emotion
            st.write(f"**Sentiment:** {primary_emotion} {emoji} (Confidence: {confidence:.2%})")
            st.write(f"**Associated Emotion:** {emotion} {emoji}")

            # Display confidence level for primary sentiment
            st.write("### Confidence Levels")
            st.progress(int(confidence * 100))
            st.write(f"{primary_emotion} {emoji}: {confidence:.2%}")

            # Display inverted/confidence distribution for other sentiments
            other_labels = [k for k in label_emotion_mapping if k != label]
            other_confidences = [(1 - confidence) / len(other_labels)] * len(other_labels)
            for i, other_label in enumerate(other_labels):
                other_emotion = label_emotion_mapping[other_label]['primary']
                other_emoji = label_emotion_mapping[other_label]['emoji']
                st.progress(int(other_confidences[i] * 100))
                st.write(f"{other_emotion} {other_emoji}: {other_confidences[i]:.2%}")

# Run app
if __name__ == "__main__":
    main_app()
