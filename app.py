import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 TechNova AI Text Summarizer")
st.write("Paste long text below and generate a concise AI summary.")

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

text = st.text_area("Enter your text here:", height=300)

if st.button("Generate Summary"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Generating summary..."):
            inputs = tokenizer(
                text,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            )

            with torch.no_grad():
                summary_ids = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=150,
                    min_length=40,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )

            summary = tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )

        original_words = len(text.split())
        summary_words = len(summary.split())

        compression_ratio = (summary_words / original_words) * 100

        st.subheader("Generated Summary")
        st.success(summary)

        st.subheader("Text Statistics")
        st.write(f"Original Length: {original_words} words")
        st.write(f"Summary Length: {summary_words} words")
        st.write(f"Reduced By: {original_words - summary_words} words")
        st.write(f"Compression Ratio: {compression_ratio:.2f}%")
        st.write(f"Content Reduced By: {100 - compression_ratio:.2f}%")