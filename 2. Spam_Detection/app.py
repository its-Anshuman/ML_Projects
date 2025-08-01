import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import re

# Download required NLTK data
nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Preprocessing function (same as used during training)
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    return ' '.join(tokens)

# Streamlit App UI
st.set_page_config(page_title="Spam Classifier", layout="centered")

st.title("📩 SMS Spam Classifier")

input_sms = st.text_area("Enter the message here:")

if st.button('Predict'):
    # 1. Preprocess
    transformed = transform_text(input_sms)

    # 2. Vectorize
    vector_input = vectorizer.transform([transformed])

    # 3. Predict
    result = model.predict(vector_input)[0]
    prob = model.predict_proba(vector_input)[0]

    # 4. Output
    if result == 1:
        st.error("❌ Spam Message")
    else:
        st.success("✅ Not Spam")

    st.write("🔍 Confidence:", f"{max(prob)*100:.2f}%")
