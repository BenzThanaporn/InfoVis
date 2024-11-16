import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Load your CRF model (replace with your actual model path)
model = joblib.load("C:\Users\Benz\Downloads\model.joblib")

stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]

def tokens_to_features(tokens, i):
    word = tokens[i]

    features = {
        "bias": 1.0,
        "word.word": word,
        "word[:3]": word[:3],
        "word.isspace()": word.isspace(),
        "word.is_stopword()": word in stopwords,
        "word.isdigit()": word.isdigit(),
        "word.islen5": word.isdigit() and len(word) == 5
    }

    if i > 0:
        prevword = tokens[i - 1]
        features.update({
            "-1.word.prevword": prevword,
            "-1.word.isspace()": prevword.isspace(),
            "-1.word.is_stopword()": prevword in stopwords,
            "-1.word.isdigit()": prevword.isdigit(),
        })
    else:
        features["BOS"] = True

    if i < len(tokens) - 1:
        nextword = tokens[i + 1]
        features.update({
            "+1.word.nextword": nextword,
            "+1.word.isspace()": nextword.isspace(),
            "+1.word.is_stopword()": nextword in stopwords,
            "+1.word.isdigit()": nextword.isdigit(),
        })
    else:
        features["EOS"] = True

    return features

def visualize_entities_crf(text: str):
    """Visualizes entities using your CRF model and HTML."""
    tokens = text.split()

    try:
        features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
        predicted_tags = model.predict([features])[0]
    except Exception as e:
        st.error(f"Error processing text: {text}\nTokens: {tokens}\nError: {e}")
        return

    # Create a Pandas DataFrame for visualization
    df = pd.DataFrame({'Token': tokens, 'Tag': predicted_tags})

    # Color mapping for entity types
    colors = {
        "O": "#5bc0de",        # Blue
        "LOC": "#ff69b4",     # Pink
        "POST": "#9370db",    # Purple
        "ADDR": "#d3d3d3"     # Grey
    }

    # Generate HTML for highlighted entities
    highlighted_text = "<div style='font-size: 16px; font-family: Arial, sans-serif;'>"
    highlighted_text += " ".join(
        [f"<span style='background-color: {colors[tag]}; padding: 5px 8px; border-radius: 8px; margin: 2px;'>{token}</span>"
         for token, tag in zip(df["Token"], df["Tag"])]
    )
    highlighted_text += "</div>"

    st.markdown(highlighted_text, unsafe_allow_html=True)

# Set up Streamlit app
st.title('Address Label Visualizer')

# Text input
text = st.text_area('Enter the address here:', value='นายมงคล 123/4 ตำบล สัตหีบ อำเภอ สัตหีบ จังหวัด ชลบุรี 20180')

# Submit button
if st.button('Visualize Labels'):
    visualize_entities_crf(text)