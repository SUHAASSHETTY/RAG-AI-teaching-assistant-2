import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from config import api_key  # Your Grok API key stored here

# Initialize Grok client
client = Groq(api_key=api_key)

# --- Streamlit Page Setup ---
st.set_page_config(page_title="RAG AI Teaching Assistant 2.0", page_icon="🎓", layout="wide")

st.markdown(
    """
    <h2 style='text-align: center; color: #FF4B4B;'>RAG AI Teaching Assistant 2.0</h2>
    <p style='text-align: center; color: gray;'>Transforming hours of video lectures into instant, intelligent answers.</p>
    <hr style='border: 1px solid #FF4B4B;'>
    """,
    unsafe_allow_html=True
)

# --- Function to Create Embedding ---
def create_embedding(text_list):
    """
    Generate embeddings using Ollama local API.
    (You can later switch this to another service if needed)
    """
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()["embeddings"]
    return embedding

# --- Function to Query Grok API ---
def inference_groq_stream(prompt):
    """
    Generate streamed response using Grok API.
    """
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    response = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content if chunk.choices[0].delta and chunk.choices[0].delta.content else ""
        response += content
    return response

# --- Load Precomputed Embeddings ---
@st.cache_resource
def load_embeddings():
    return joblib.load("embeddings.joblib")  # or new merged file if renamed

df = load_embeddings()

# --- Input Section ---
st.subheader("Ask your question 📚")
incoming_query = st.text_input("Example: 'Where are form and input tags taught in the course?'")

if st.button("Ask"):
    if incoming_query.strip() == "":
        st.warning("Please enter a question before proceeding.")
    else:
        with st.spinner("Analyzing content..."):
            # Generate embedding for the query
            question_embedding = create_embedding([incoming_query])[0]

            # Compute similarity
            similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
            top_results = 5
            max_indx = similarities.argsort()[::-1][0:top_results]
            new_df = df.loc[max_indx]

            # Build prompt for Grok
            prompt = f"""
            You are an AI teaching assistant for a Python Web Development course.
            Below are video subtitle chunks with titles, video numbers, start, end, and text:
            {new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
            ---------------------------------
            Question: "{incoming_query}"
            Answer naturally — explain where this topic is taught, mention video number and timestamps,
            and guide the user to go to that particular video.
            If the question is unrelated to the course, politely say you can only answer course-related questions.
            """

            # Generate and display Grok response
            answer = inference_groq_stream(prompt)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("🎯 AI Assistant’s Answer")
        st.write(answer)

        # Optionally show relevant chunks
        with st.expander("View retrieved video chunks"):
            st.dataframe(new_df[["title", "number", "start", "end", "text"]])

st.markdown(
    """
    <hr>
    <p style='text-align: center; color: gray;'>Developed by <b>Suhaas S</b> | IIIT Dharwad</p>
    """,
    unsafe_allow_html=True
)
