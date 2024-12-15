import streamlit as st
import pandas as pd
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer
import re
from load_model import translate_T5, translate_BERT_BARTPho, translate_LSTM, translate_BiLSTM, translate_GRU, translate_MarianMT
# from load_model import translate_GRU
import sentencepiece as spm


# python -m streamlit run ui.py

MAX_LENGTH = 64

st.title("Machine Translation")
st.markdown('<p style="font-size:24px; font-weight:bold;">English - Vietnamese</p>', 
            unsafe_allow_html=True)

if 'summarize_model' not in st.session_state:
    summarize_model_dir = "./Summarization"
    st.session_state.summarize_tokenizer = AutoTokenizer.from_pretrained(summarize_model_dir)
    st.session_state.summarize_model = AutoModelForSeq2SeqLM.from_pretrained(summarize_model_dir)
    print("Summarize model loaded")
    
model_name = st.selectbox("Select Model", ["BERT_BARTPho" ,"T5", "BiLSTM", "GRU", "LSTM", "MarianMT"], index=None, placeholder="Select a Model")

input_text = st.text_area(
    "Input Text:",
    placeholder="Enter your text here...",
    height=150,
    key="input_text",
    help = f"If your input text is more than {MAX_LENGTH} words. It will be summarized and then translated",
    value= "Today, I go to school"
)

def summarize(input_text):
    if (len(input_text.split()) > MAX_LENGTH):
        st.write("Your input paragraph is more than 64 words!")
        
        summarize_tokenizer = st.session_state.summarize_tokenizer
        summarize_model = st.session_state.summarize_model
        
        inputs = summarize_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = summarize_model.generate(**inputs, max_length=100, num_beams=5, length_penalty=2.0, early_stopping=True)
        
        summerized_input_text = summarize_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return summerized_input_text

def cut_sentence(input_text):
    sentences = re.split(r'(?<=[.!?]) +', input_text.strip())
    return sentences

st.write(summarize(input_text))

st.markdown(
    f"""
    <style>
    input[type=text] {{
        width: 500%;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

if st.button("Translate"):
    if model_name == "BERT_BARTPho":
        translated_text = translate_BERT_BARTPho(input_text)
        st.write(f"Translation for {model_name}:")
        st.write(translated_text)
        
    if model_name == "T5":
        translated_text = translate_T5(input_text)
        st.write(f"Translation for {model_name}:")
        st.write(translated_text)
    
    if model_name == "BiLSTM":
        translated_text = translate_BiLSTM(input_text)
        st.write(f"Translation for {model_name}:")
        st.write(translated_text)

    elif model_name == "GRU":
        translated_text = translate_GRU(input_text)
        st.write(f"Translation for {model_name}:")
        st.write(translated_text)
            
    elif model_name == "LSTM":
        translated_text = translate_LSTM(input_text)
        st.write(f"Translation for {model_name}:")
        st.write(translated_text)
        
    elif model_name == "MarianMT":
        translated_text = translate_MarianMT(input_text)
        st.write(f"Translation for {model_name}:")
        st.write(translated_text)

