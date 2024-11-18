import streamlit as st
import pandas as pd
import networkx as nx
from wordcloud import WordCloud
from pyvis.network import Network
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.font_manager as fm
import joblib
import math
import random
from collections import defaultdict

# Load the CSV file with ID column
data = pd.read_csv('thai_addresses_with_ID(2).csv')

# Load your CRF model (replace with your actual model path)
model = joblib.load('model.joblib')

# Initialize word_accumulator if not in session state
if "word_accumulator" not in st.session_state:
    st.session_state.word_accumulator = defaultdict(lambda: defaultdict(int))

# Use st.session_state to retain previous word counts
word_accumulator = st.session_state.word_accumulator

# Define the default sentences for NER
default_sentences = [
    'นายสมชาย เข็มกลัด 254 ถนน พญาไท แขวง วังใหม่ เขต ปทุมวัน กรุงเทพมหานคร 10330',
    'นายมงคล 123/4 ตำบล บ้านไกล อำเภอ เมือง จังหวัด ลพบุรี 15000'
]

# Function to extract features for each token
def tokens_to_features(tokens, i):
    word = tokens[i]
    stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]

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

# Function to visualize entities using CRF
def visualize_entities_crf(text: str):
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
        "O": "#6699CC",       # Dark Blue
        "LOC": "#FF7F00",     # Medium Orange
        "POST": "#FF3399",    # Bright Pink
        "ADDR": "#CCCCCC"     # Grey
    }

    # Generate HTML for highlighted entities
    highlighted_text = "<div style='font-size: 16px;'>"
    highlighted_text += " ".join(
        [f"<span style='background-color: {colors[tag]}; padding: 2px 5px;'>{token}</span>"
        for token, tag in zip(df["Token"], df["Tag"])]
    )
    highlighted_text += "</div>"


    # Display the highlighted text in Streamlit
    st.markdown(highlighted_text, unsafe_allow_html=True)

# Sidebar for Page Navigation
st.sidebar.title("Explore the Analysis")
page = st.sidebar.selectbox("Select Page", ["Introduction", "Word Clouds", "Tag Distribution Comparison", "Word Relationships", "Conclusions & Insights"])

# Introduction Page
if page == "Introduction":
    st.title("Model Bias Analysis for Location Prediction")
    st.markdown("""
    Welcome to the analysis of the model's performance in predicting locations as either `Bangkok` or `Provincial Areas`.
    This tool will guide you through different visualizations that explore whether there's a **bias** in how the model predicts tags for different locations.
    
    Let's begin by exploring the data through **Word Clouds**, followed by **Tag Distribution Comparisons** between Bangkok and other provinces, and finally dive into **relationships** between words to see deeper connections.
    
    Use the sidebar to navigate between the different parts of the analysis.
    """)

    # NER Visualization Section
    st.header("Named Entity Recognition (NER) Visualization")
    if "sentence_index" not in st.session_state:
        st.session_state.sentence_index = 0

    current_sentence_index = st.session_state.sentence_index

    if current_sentence_index < len(default_sentences):
        st.write(f"### Default Sentence {current_sentence_index + 1}")
        st.write(default_sentences[current_sentence_index])
        if st.button("Next Sentence"):
            st.session_state.sentence_index += 1
    else:
        st.write("### Enter your own address below:")
        user_text = st.text_area('Enter the address here:', value='นายมงคล 123/4 ตำบล บ้านไกล อำเภอ เมือง จังหวัด ลพบุรี 15000')
        if st.button('Visualize Labels'):
            visualize_entities_crf(user_text)

    if current_sentence_index < len(default_sentences):
        visualize_entities_crf(default_sentences[current_sentence_index])


# Word Clouds Page
elif page == "Word Clouds":
    st.title("Word Clouds by Tag")
    st.markdown("""
    Word clouds are a great way to visualize the most frequent words that the model categorized under different tags.
    Let's see if there are any differences in which words are tagged as `LOC`, `ADDR`, `POST`, and `O` for Bangkok and other provinces.
    """)

    # Filtering Options
    st.sidebar.header('Filtering Options')
    selected_tags = st.sidebar.multiselect(
        'Select Tags to Display', ['LOC', 'ADDR', 'POST', 'O'], ['LOC', 'ADDR', 'POST', 'O']
    )

    all_button = st.sidebar.button('ทั้งหมด')
    bangkok_button = st.sidebar.button('กรุงเทพ')
    province_button = st.sidebar.button('ต่างจังหวัด')

    if bangkok_button:
        selected_min_id, selected_max_id = 1, 25
    elif province_button:
        selected_min_id, selected_max_id = 26, 50
    elif all_button or not (bangkok_button or province_button):
        selected_min_id, selected_max_id = data['ID'].min(), data['ID'].max()

    filtered_data = data[(data['ID'] >= selected_min_id) & (data['ID'] <= selected_max_id)]

    wordcloud_colors = {
        'LOC': "#FF7F00",    # Medium Orange
        'ADDR': "#CCCCCC",   # Grey
        'POST': "#FF3399",   # Bright Pink
        'O': "#6699CC"       # Dark Blue
    }

    # Create Word Clouds for each selected tag
    categories = {
        'LOC': filtered_data[filtered_data['Predicted Tag'] == 'LOC']['Token'],
        'ADDR': filtered_data[filtered_data['Predicted Tag'] == 'ADDR']['Token'],
        'POST': filtered_data[filtered_data['Predicted Tag'] == 'POST']['Token'],
        'O': filtered_data[filtered_data['Predicted Tag'] == 'O']['Token']
    }

    for category, words in categories.items():
        if len(words) > 0 and category in selected_tags:
            word_freq = Counter(words)
            
            # Normalize word frequencies to assign consistent sizes for words with the same frequency
            max_freq = max(word_freq.values())
            normalized_sizes = {word: int((freq / max_freq) * 100) for word, freq in word_freq.items()}

            # Generate word cloud using normalized sizes
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                color_func=lambda *args, **kwargs: wordcloud_colors[category],
                font_path='Sarabun-Regular.ttf'
            ).generate_from_frequencies(normalized_sizes)
            
            # Plot the word cloud
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'{category} Word Cloud')
            st.pyplot(plt)



# Tag Distribution Comparison Page
elif page == "Tag Distribution Comparison":
    st.title("Tag Distribution: Bangkok vs. Other Provinces")
    st.markdown("""
    Let's compare how the tags are distributed between `Bangkok` and other provinces.
    This comparison can help identify whether the model shows any signs of **bias** when predicting tags for different regions.
    """)

    # Filter data for Bangkok and other provinces
    bangkok_data = data[(data['ID'] >= 1) & (data['ID'] <= 25)]
    province_data = data[(data['ID'] >= 26) & (data['ID'] <= 50)]

    # Count frequency of each tag for Bangkok and other provinces
    bangkok_tag_counts = bangkok_data['Predicted Tag'].value_counts()
    province_tag_counts = province_data['Predicted Tag'].value_counts()

    bar_colors = {
        'LOC': {'bangkok': '#FF7F00', 'province': '#FFCC99'},  # Medium Orange, Light Orange
        'ADDR': {'bangkok': '#555555', 'province': '#CCCCCC'},  # Dark Grey, Light Grey
        'POST': {'bangkok': '#FF3399', 'province': '#FF99CC'},  # Bright Pink, Light Pink
        'O': {'bangkok': '#003366', 'province': '#6699CC'}  # Dark Blue, Light Blue
    }

    # Create a bar chart comparing tag distributions
    fig, ax = plt.subplots()
    tags = ['LOC', 'ADDR', 'POST', 'O']

    # Get counts for Bangkok and provinces
    bangkok_counts = [bangkok_tag_counts.get(tag, 0) for tag in tags]
    province_counts = [province_tag_counts.get(tag, 0) for tag in tags]

    bar_width = 0.35
    index = range(len(tags))

    # Plot bars with specific colors
    for i, tag in enumerate(tags):
        ax.bar(index[i], bangkok_counts[i], bar_width, 
            label=f'Bangkok - {tag}' if i == 0 else "", 
            color=bar_colors[tag]['bangkok'])
        ax.bar(index[i] + bar_width, province_counts[i], bar_width, 
            label=f'Province - {tag}' if i == 0 else "", 
            color=bar_colors[tag]['province'])

    # Set chart labels and legend
    ax.set_xlabel('Tags')
    ax.set_ylabel('Frequency')
    ax.set_title('Tag Frequency Comparison: Bangkok vs. Other Provinces')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(tags)
    ax.legend(loc='upper right')

    # Display in Streamlit
    st.pyplot(fig)


# Word Relationships Page
elif page == "Word Relationships":
    st.title("Word Relationships Between Tokens")
    st.markdown("""
    Here, we explore the relationships between words within each sentence to understand potential **contextual connections**.
    This graph is interactive and allows you to explore how tokens relate to one another.
    """)

    G = nx.Graph()
    # (Graph creation code remains the same)
    # ... Add nodes and edges to the graph ...
    net = Network(notebook=False, height='700px', width='100%', bgcolor='#ffffff', font_color='#000000')
    net.from_nx(G)
    net.save_graph('filtered_word_relationship_graph.html')
    st.components.v1.html(open('filtered_word_relationship_graph.html', 'r').read(), height=700, scrolling=True)

# Conclusions & Insights Page
elif page == "Conclusions & Insights":
    st.title("Conclusions & Insights")
    st.markdown("""
    ### Summary of Findings:
    - The **Word Clouds** provided an overview of the most frequent tokens categorized by the model.
    - The **Tag Distribution Comparison** revealed potential biases in tag predictions between `Bangkok` and `Other Provinces`.
    - The **Word Relationships** visualized contextual connections between tokens, offering insight into how the model might be interpreting relationships.

    ### Next Steps:
    - To address observed biases, consider expanding the dataset to include more examples from underrepresented regions.
    - Further **tuning** of the model might also help in achieving more balanced predictions.
    """)
