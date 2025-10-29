# RAG based AI Assistant - Advanced Professional Theme with Groq API

import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
import json
import os
from datetime import datetime
import time
from groq import Groq

# Import Groq API key from config
try:
    from config import api_key as groq_api_key
    groq_client = Groq(api_key=groq_api_key)
except ImportError:
    st.error("Please create a config.py file with your Groq API key: api_key = 'your_key_here'")
    groq_client = None

# Page configuration
st.set_page_config(
    page_title="RAG AI Teaching Assistant",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS with Black and White/Red Theme
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: #000000;
        color: #ffffff;
    }
    
    /* Main Header */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: 2px;
        border-bottom: 3px solid #dc143c;
        padding-bottom: 1rem;
    }
    
    .sub-header {
        text-align: center;
        color: #dc143c;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Cards */
    .metric-card {
        background: #1a1a1a;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #333333;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #dc143c;
        box-shadow: 0 6px 20px rgba(220, 20, 60, 0.3);
    }
    
    /* Result Cards */
    .result-card {
        background: #1a1a1a;
        padding: 2rem;
        border-radius: 0.5rem;
        border: 2px solid #333333;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
        color: #ffffff;
        line-height: 1.6;
    }
    
    .result-card:hover {
        border-color: #dc143c;
    }
    
    /* Timestamp Badges */
    .timestamp-badge {
        background: #dc143c;
        color: #ffffff;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        transition: all 0.3s ease;
    }
    
    .timestamp-badge:hover {
        background: #ff1744;
    }
    
    /* Similarity Badges */
    .similarity-badge-high {
        background: #ffffff;
        color: #000000;
        padding: 0.4rem 0.8rem;
        border-radius: 0.3rem;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 2px solid #ffffff;
    }
    
    .similarity-badge-medium {
        background: #dc143c;
        color: #ffffff;
        padding: 0.4rem 0.8rem;
        border-radius: 0.3rem;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 2px solid #dc143c;
    }
    
    .similarity-badge-low {
        background: transparent;
        color: #ffffff;
        padding: 0.4rem 0.8rem;
        border-radius: 0.3rem;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 2px solid #ffffff;
    }
    
    /* Status Messages */
    .status-success {
        background: #1a1a1a;
        border-left: 4px solid #ffffff;
        padding: 1rem;
        border-radius: 0.3rem;
        color: #ffffff;
        margin: 1rem 0;
    }
    
    .status-error {
        background: #1a1a1a;
        border-left: 4px solid #dc143c;
        padding: 1rem;
        border-radius: 0.3rem;
        color: #dc143c;
        margin: 1rem 0;
    }
    
    .status-warning {
        background: #1a1a1a;
        border-left: 4px solid #ffa500;
        padding: 1rem;
        border-radius: 0.3rem;
        color: #ffa500;
        margin: 1rem 0;
    }
    
    /* Custom Buttons */
    .stButton>button {
        background: #dc143c;
        color: #ffffff;
        border: none;
        border-radius: 0.3rem;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        background: #ff1744;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(220, 20, 60, 0.4);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #0a0a0a;
        border-right: 2px solid #333333;
    }
    
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #dc143c;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.9rem;
    }
    
    /* Text Input Areas */
    .stTextArea textarea {
        background: #1a1a1a;
        color: #ffffff;
        border: 2px solid #333333;
        border-radius: 0.3rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #dc143c;
        box-shadow: 0 0 10px rgba(220, 20, 60, 0.2);
    }
    
    /* Text Input */
    .stTextInput input {
        background: #1a1a1a;
        color: #ffffff;
        border: 2px solid #333333;
        border-radius: 0.3rem;
    }
    
    .stTextInput input:focus {
        border-color: #dc143c;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #1a1a1a;
        border: 1px solid #333333;
        border-radius: 0.3rem;
        color: #ffffff;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #dc143c;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #dc143c;
        font-size: 2rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #999999;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: #dc143c;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #1a1a1a;
        border: 2px solid #333333;
        border-radius: 0.3rem;
        color: #ffffff;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: #dc143c;
        border-color: #dc143c;
        color: #ffffff;
    }
    
    /* Section Headers */
    h3 {
        color: #dc143c;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 2px solid #333333;
        padding-bottom: 0.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #999999;
        padding: 2rem;
        border-top: 2px solid #333333;
        margin-top: 3rem;
        background: #0a0a0a;
        border-radius: 0.5rem;
    }
    
    .footer h4 {
        color: #dc143c;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: #1a1a1a;
        border: 2px solid #333333;
        color: #ffffff;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: #dc143c;
    }
    
    /* Multiselect */
    .stMultiSelect > div > div {
        background: #1a1a1a;
        border: 2px solid #333333;
    }
    
    .stMultiSelect [data-baseweb="tag"] {
        background: #dc143c;
        color: #ffffff;
    }
    
    .stMultiSelect input {
        color: #ffffff !important;
    }
    
    .stMultiSelect [role="button"] {
        color: #dc143c;
    }
    
    /* Info/Warning boxes */
    .stAlert {
        background: #1a1a1a;
        border: 1px solid #333333;
        color: #ffffff;
    }
    
    /* General text color */
    p, span, div {
        color: #ffffff;
    }
    
    /* Links */
    a {
        color: #dc143c;
    }
    
    a:hover {
        color: #ff1744;
    }
    
    /* Download button */
    .stDownloadButton button {
        background: #dc143c !important;
        color: #ffffff !important;
    }
    
    .stDownloadButton button:hover {
        background: #ff1744 !important;
    }
    
    .stDownloadButton button p,
    .stDownloadButton button span,
    .stDownloadButton button div {
        color: #ffffff !important;
    }
    
    /* DataFrame styling */
    .dataframe {
        background: #1a1a1a;
        color: #ffffff;
    }
    
    /* Info boxes */
    .info-box {
        background: #1a1a1a;
        border: 2px solid #333333;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box:hover {
        border-color: #dc143c;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'last_results' not in st.session_state:
    st.session_state.last_results = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

@st.cache_data
def load_embeddings():
    """Load the pre-computed embeddings (merged chunks - 5 segments combined)"""
    try:
        if os.path.exists('embeddings.joblib'):
            df = joblib.load('embeddings.joblib')
            return df
        else:
            st.error("embeddings.joblib file not found. Please run merge_chunks.py and generate embeddings first.")
            return None
    except Exception as e:
        st.error(f"Error loading embeddings: {str(e)}")
        return None

def create_embedding(text_list):
    """Create embeddings using Ollama API with bge-m3 model"""
    try:
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": text_list
        }, timeout=30)
        r.raise_for_status()
        embedding = r.json()["embeddings"] 
        return embedding
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to Ollama. Make sure Ollama is running: ollama serve")
        return None
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None

def generate_response_groq(prompt, model="llama-3.3-70b-versatile", temperature=0.7, max_tokens=2048):
    """Generate response using Groq API"""
    if groq_client is None:
        st.error("Groq client not initialized. Please check your API key in config.py")
        return None
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI teaching assistant for a Web Development course. Provide clear, accurate answers based on the video content provided, and always reference specific video numbers and timestamps. Be conversational but professional."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response from Groq: {str(e)}")
        return None

def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def search_similar_content(query, df, top_k=5):
    """Search for similar content based on query embedding"""
    if df is None or df.empty:
        return None
    
    query_embedding = create_embedding([query])
    if query_embedding is None:
        return None
    
    query_embedding = query_embedding[0]
    
    embeddings_matrix = np.vstack(df['embedding'].values)
    similarities = cosine_similarity(embeddings_matrix, [query_embedding]).flatten()
    
    top_indices = similarities.argsort()[::-1][:top_k]
    results = df.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]
    
    return results

def check_ollama_connection():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_groq_connection():
    """Check if Groq API is accessible"""
    if groq_client is None:
        return False
    try:
        groq_client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model="llama-3.3-70b-versatile",
            max_tokens=5,
        )
        return True
    except:
        return False

def export_to_markdown(query, ai_response, results):
    """Export results to markdown format"""
    md_content = f"""# Video Q&A Search Results
    
**Query:** {query}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## AI Response

{ai_response}

---

## Relevant Content Chunks (Merged)

Note: Each chunk represents approximately 5 merged subtitle segments for better context.

"""
    for idx, row in results.iterrows():
        start_time = format_timestamp(row.get('start', 0))
        end_time = format_timestamp(row.get('end', 0))
        md_content += f"""
### {row.get('title', 'Unknown')} - Video {row.get('number', 'N/A')}
- **Timestamp:** {start_time} - {end_time}
- **Similarity Score:** {row['similarity']:.4f}
- **Content:** {row.get('text', 'No text available')}

---
"""
    return md_content

def get_similarity_badge_class(similarity):
    """Return appropriate badge class based on similarity score"""
    if similarity > 0.7:
        return "similarity-badge-high"
    elif similarity > 0.5:
        return "similarity-badge-medium"
    else:
        return "similarity-badge-low"

def main():
    # Header
    st.markdown('<h1 class="main-header">RAG BASED AI TEACHING ASSISTANT</h1>', unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### SYSTEM CONFIGURATION")
        
        # Connection Status
        ollama_connected = check_ollama_connection()
        groq_connected = check_groq_connection()
        
        st.markdown("#### Connection Status")
        if ollama_connected:
            st.markdown('<div class="status-success">OLLAMA: Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">OLLAMA: Disconnected</div>', unsafe_allow_html=True)
            st.info("Start Ollama: `ollama serve`")
        
        if groq_connected:
            st.markdown('<div class="status-success">GROQ API: Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">GROQ API: Not Connected</div>', unsafe_allow_html=True)
            st.info("Check API key in config.py")
        
        st.markdown("---")
        
        # Model Configuration
        st.markdown("#### Groq Model Settings")
        
        groq_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        
        selected_model = st.selectbox(
            "Select Model", 
            groq_models, 
            index=0,
            help="Choose the Groq model for response generation"
        )
        
        temperature = st.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7, 
            step=0.1,
            help="""Controls response randomness:
• 0.0-0.3: Very focused, deterministic, factual answers
• 0.4-0.7: Balanced creativity and accuracy (recommended)
• 0.8-1.0: More creative, varied, but less predictable responses"""
        )
        
        max_tokens = st.slider(
            "Max Tokens", 
            min_value=512, 
            max_value=4096, 
            value=2048, 
            step=512,
            help="""Maximum response length:
• 512: Short, concise answers (~400 words)
• 1024: Medium length responses (~800 words)
• 2048: Detailed explanations (~1600 words)
• 4096: Very comprehensive responses (~3200 words)
Note: 1 token ≈ 0.75 words"""
        )
        
        st.markdown("---")
        
        # Search Configuration
        st.markdown("#### Search Settings")
        
        top_k = st.slider(
            "Number of Results", 
            min_value=1, 
            max_value=15, 
            value=5,
            help="Number of similar chunks to retrieve"
        )
        
        min_similarity = st.slider(
            "Minimum Similarity", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.0, 
            step=0.05,
            help="Filter results below this threshold"
        )
        
        st.markdown("---")
        
        # Display Options
        st.markdown("#### Display Settings")
        
        show_full_text = st.checkbox("Show Full Text", value=True)
        show_timestamps = st.checkbox("Show Timestamps", value=True)
        show_similarity = st.checkbox("Show Similarity Scores", value=True)
        
        st.markdown("---")
        
        # Statistics
        st.markdown("#### Session Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", st.session_state.query_count)
        with col2:
            st.metric("History", len(st.session_state.search_history))
        
        if st.button("Clear History", use_container_width=True):
            st.session_state.search_history = []
            st.session_state.query_count = 0
            st.session_state.last_results = None
            st.session_state.last_query = ""
            st.rerun()
        
        st.markdown("---")
        
        # Recent Searches
        if st.session_state.search_history:
            st.markdown("#### Recent Searches")
            for i, hist in enumerate(reversed(st.session_state.search_history[-5:])):
                with st.expander(f"Query {len(st.session_state.search_history) - i}"):
                    st.text(hist['query'][:80] + "..." if len(hist['query']) > 80 else hist['query'])
                    st.caption(f"Time: {hist['timestamp']}")
                    st.caption(f"Results: {hist.get('results_count', 0)}")
        
        st.markdown("---")
        
        # System Information
        with st.expander("System Information"):
            st.markdown("""
            **Technology Stack:**
            - Embeddings: BGE-M3 (Ollama)
            - LLM: Groq API
            - Chunks: Merged (5 segments)
            - Similarity: Cosine Similarity
            - Framework: Streamlit
            """)
    
    # Load embeddings
    with st.spinner("Loading merged embeddings database..."):
        df = load_embeddings()
    
    if df is None:
        st.stop()
    
    # Database Overview
    st.markdown("### DATABASE OVERVIEW")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Merged Chunks", f"{len(df):,}", help="Total number of merged chunks (5 segments each)")
    
    with col2:
        unique_videos = df['title'].nunique() if 'title' in df.columns else 0
        st.metric("Unique Videos", unique_videos, help="Number of distinct video titles")
    
    with col3:
        if 'end' in df.columns and 'start' in df.columns:
            # Calculate actual duration: max(end) - min(start) for total video length
            total_duration = (df['end'].max() - df['start'].min()) / 60
        else:
            total_duration = 0
        st.metric("Total Duration", f"{total_duration:.1f} min", help="Total video content duration")
    
    with col4:
        avg_chunk_length = df['text'].str.len().mean() if 'text' in df.columns else 0
        st.metric("Avg Chunk Length", f"{avg_chunk_length:.0f} chars", help="Average characters per merged chunk")
    
    st.markdown("---")
    
    # Main Query Interface
    st.markdown("### SEARCH INTERFACE")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Simple Search", "Advanced Filters", "Batch Query"])
    
    with tab1:
        st.markdown("#### Ask Your Question")
        
        query = st.text_area(
            "Enter your question about the course:",
            placeholder="Example: Where are form and input tags taught in the course?",
            height=120,
            key="main_query",
            help="Ask any question about the video course content"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_clicked = st.button("Search", type="primary", use_container_width=True)
        
        with col2:
            if st.button("Example Query", use_container_width=True):
                st.session_state.main_query = "How to create a basic website structure?"
                st.rerun()
        
        with col3:
            if st.button("Clear", use_container_width=True):
                st.session_state.main_query = ""
                st.rerun()
        
        # Example queries
        with st.expander("Example Questions"):
            st.markdown("""
            - Where is HTML introduction covered?
            - How to setup VS Code for web development?
            - Which video explains CSS styling?
            - Where are JavaScript alerts discussed?
            - How to create forms in HTML?
            - Where is the bookmark manager project explained?
            """)
    
    with tab2:
        st.markdown("#### Filter by Video Properties")
        
        filter_number = st.multiselect(
            "Select Video Numbers",
            options=sorted(df['number'].unique().tolist()) if 'number' in df.columns else [],
            default=None,
            help="Filter by specific video numbers (e.g., Video 1, Video 2)"
        )
        
        advanced_query = st.text_area(
            "Enter your filtered question:",
            placeholder="Ask a question within the filtered videos...",
            height=100,
            key="advanced_query",
            help="This query will only search within the filtered videos"
        )
        
        advanced_search = st.button("Search with Filters", type="primary", use_container_width=True)
    
    with tab3:
        st.markdown("#### Batch Query Processing")
        st.info("Enter multiple questions (one per line) to process them together")
        
        batch_queries = st.text_area(
            "Enter multiple questions:",
            placeholder="How to setup VS Code?\nWhere is CSS explained?\nWhich video covers forms?",
            height=150,
            key="batch_queries"
        )
        
        batch_search = st.button("Process Batch", type="primary", use_container_width=True)
    
    # Handle search operations
    if (search_clicked and query.strip()) or (advanced_search and advanced_query.strip()):
        current_query = query.strip() if search_clicked else advanced_query.strip()
        
        # Apply filters if in advanced mode
        filtered_df = df.copy()
        if advanced_search:
            if filter_number:
                filtered_df = filtered_df[filtered_df['number'].isin(filter_number)]
            
            if filtered_df.empty:
                st.warning("No chunks match the selected filters. Try different video numbers.")
                st.stop()
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Create embedding
            status_text.text("Step 1/4: Creating query embedding...")
            progress_bar.progress(25)
            time.sleep(0.3)
            
            # Step 2: Search
            status_text.text("Step 2/4: Searching for relevant content...")
            results = search_similar_content(current_query, filtered_df, top_k)
            progress_bar.progress(50)
            time.sleep(0.3)
            
            if results is not None and not results.empty:
                # Filter by minimum similarity
                results = results[results['similarity'] >= min_similarity]
                
                if results.empty:
                    st.warning(f"No results found with similarity score >= {min_similarity}. Try lowering the threshold.")
                    progress_bar.empty()
                    status_text.empty()
                else:
                    # Step 3: Generate response
                    status_text.text("Step 3/4: Generating AI response with Groq...")
                    progress_bar.progress(75)
                    
                    # Update statistics
                    st.session_state.query_count += 1
                    st.session_state.search_history.append({
                        'query': current_query,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'results_count': len(results)
                    })
                    
                    # Prepare context for Groq
                    context_data = results[["title", "number", "start", "end", "text"]].to_json(orient="records")
                    
                    prompt = f'''I am teaching Web Development in my Sigma course. Here are relevant video subtitle chunks (each chunk is a merge of approximately 5 consecutive segments for better context):

{context_data}

---------------------------------
User Question: "{current_query}"
---------------------------------

Instructions:
Provide a CLEAR, STRUCTURED answer in this EXACT format:

Covered in:

Video [number]: "[title]"

1. [start_time] - [end_time]:
   * Concept: [Brief concept name]
   * Explanation: [Clear 1-2 sentence explanation of what's covered]

2. [start_time] - [end_time]:
   * Concept: [Brief concept name]
   * Explanation: [Clear 1-2 sentence explanation of what's covered]

(Repeat for each relevant timestamp)

Video [number]: "[title]" (if multiple videos)

1. [start_time] - [end_time]:
   * Concept: [Brief concept name]
   * Explanation: [Clear 1-2 sentence explanation]

IMPORTANT RULES:
- Use numbered lists (1, 2, 3...) for each timestamp section
- Each entry MUST have both "Concept" and "Explanation"
- Keep explanations clear and concise (1-2 sentences maximum)
- Convert seconds to MM:SS format (e.g., 1127.8 becomes 18:47)
- Group by video number with video title
- If topic not found, state: "This topic is not covered in the available course content."

Please provide your response:'''

                    ai_response = generate_response_groq(prompt, selected_model, temperature, max_tokens)
                    
                    # Step 4: Complete
                    status_text.text("Step 4/4: Finalizing results...")
                    progress_bar.progress(100)
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Store results
                    st.session_state.last_results = results
                    st.session_state.last_query = current_query
                    
                    if ai_response:
                        # Display AI Response
                        st.markdown(f"""
                        <div class="result-card">
                            {ai_response}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Search Metrics
                        st.markdown("---")
                        st.markdown("### SEARCH METRICS")
                        
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        with metric_col1:
                            avg_sim = results['similarity'].mean()
                            st.metric("Average Similarity", f"{avg_sim:.4f}")
                        
                        with metric_col2:
                            max_sim = results['similarity'].max()
                            st.metric("Maximum Similarity", f"{max_sim:.4f}")
                        
                        with metric_col3:
                            st.metric("Results Found", len(results))
                        
                        with metric_col4:
                            min_sim = results['similarity'].min()
                            st.metric("Minimum Similarity", f"{min_sim:.4f}")
                        
                        # Relevant Content Display
                        st.markdown("---")
                        st.markdown("### RELEVANT CONTENT CHUNKS")
                        st.caption("Note: Each chunk represents approximately 5 merged subtitle segments for enhanced context")
                        
                        for idx, row in results.iterrows():
                            badge_class = get_similarity_badge_class(row['similarity'])
                            
                            # Determine if this is the first result
                            is_first = (idx == results.index[0])
                            
                            with st.expander(
                                f"Video {row.get('number', 'N/A')}: {row.get('title', 'Unknown')} | Similarity: {row['similarity']:.4f}",
                                expanded=is_first
                            ):
                                content_col, meta_col = st.columns([3, 1])
                                
                                with content_col:
                                    st.markdown("**Merged Content:**")
                                    if show_full_text:
                                        st.write(row.get('text', 'No text available'))
                                    else:
                                        text = row.get('text', 'No text available')
                                        st.write(text[:200] + "..." if len(text) > 200 else text)
                                
                                with meta_col:
                                    if show_timestamps:
                                        start_time = format_timestamp(row.get('start', 0))
                                        end_time = format_timestamp(row.get('end', 0))
                                        st.markdown(f"""
                                        <div class="timestamp-badge">
                                            {start_time} - {end_time}
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    if show_similarity:
                                        st.markdown("<br>", unsafe_allow_html=True)
                                        st.markdown(f"""
                                        <div class="{badge_class}">
                                            Score: {row['similarity']:.4f}
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    st.markdown("<br>", unsafe_allow_html=True)
                                    st.markdown(f"**Video:** {row.get('number', 'N/A')}")
                                    st.markdown(f"**Title:** {row.get('title', 'Unknown')}")
                        
                        # Export Options
                        st.markdown("---")
                        st.markdown("### EXPORT RESULTS")
                        
                        export_col1, export_col2, export_col3 = st.columns(3)
                        
                        with export_col1:
                            st.download_button(
                                label="Download AI Response (TXT)",
                                data=ai_response,
                                file_name=f"ai_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with export_col2:
                            csv_data = results[['title', 'number', 'start', 'end', 'text', 'similarity']].to_csv(index=False)
                            st.download_button(
                                label="Download Results (CSV)",
                                data=csv_data,
                                file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with export_col3:
                            md_data = export_to_markdown(current_query, ai_response, results)
                            st.download_button(
                                label="Download Report (MD)",
                                data=md_data,
                                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown",
                                use_container_width=True
                            )
                    else:
                        st.error("Failed to generate AI response. Please check Groq API connection and try again.")
            else:
                progress_bar.empty()
                status_text.empty()
                st.error("Failed to retrieve search results. Please check Ollama connection.")
        
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"An error occurred during search: {str(e)}")
    
    elif (search_clicked or advanced_search):
        st.warning("Please enter a question before searching.")
    
    # Handle batch queries
    if batch_search and batch_queries.strip():
        queries_list = [q.strip() for q in batch_queries.strip().split('\n') if q.strip()]
        
        if not queries_list:
            st.warning("Please enter at least one question.")
        else:
            st.markdown("---")
            st.markdown("### BATCH PROCESSING RESULTS")
            st.info(f"Processing {len(queries_list)} queries...")
            
            batch_progress = st.progress(0)
            
            for i, batch_query in enumerate(queries_list):
                batch_progress.progress((i + 1) / len(queries_list))
                
                st.markdown(f"#### Query {i+1}: {batch_query}")
                
                with st.spinner(f"Processing query {i+1}..."):
                    results = search_similar_content(batch_query, df, top_k=3)
                    
                    if results is not None and not results.empty:
                        results = results[results['similarity'] >= min_similarity]
                        
                        if not results.empty:
                            context_data = results[["title", "number", "start", "end", "text"]].to_json(orient="records")
                            
                            prompt = f'''Based on these video chunks: {context_data}
                            
Question: "{batch_query}"

Provide a brief answer with video reference:'''

                            ai_response = generate_response_groq(prompt, selected_model, 0.5, 1024)
                            
                            if ai_response:
                                st.markdown(f"""
                                <div class="result-card">
                                    {ai_response}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                with st.expander("View Top Result"):
                                    top_result = results.iloc[0]
                                    st.markdown(f"**Video {top_result['number']}: {top_result['title']}**")
                                    st.markdown(f"**Timestamp:** {format_timestamp(top_result['start'])} - {format_timestamp(top_result['end'])}")
                                    st.markdown(f"**Similarity:** {top_result['similarity']:.4f}")
                            else:
                                st.warning(f"Could not generate response for query {i+1}")
                        else:
                            st.warning(f"No relevant results found for query {i+1}")
                    else:
                        st.error(f"Search failed for query {i+1}")
                
                st.markdown("---")
            
            batch_progress.empty()
            st.success(f"Completed processing {len(queries_list)} queries!")
    
    # Data Explorer Section
    st.markdown("---")
    st.markdown("### DATA EXPLORER")
    
    explorer_tab1, explorer_tab2, explorer_tab3 = st.tabs(["Video Statistics", "Content Browser", "Raw Data"])
    
    with explorer_tab1:
        st.markdown("#### Video-wise Statistics")
        
        if 'title' in df.columns and 'number' in df.columns:
            video_stats = df.groupby(['number', 'title']).agg({
                'text': 'count',
                'start': 'min',
                'end': 'max'
            }).reset_index()
            
            video_stats.columns = ['Video Number', 'Title', 'Chunk Count', 'Start Time', 'End Time']
            video_stats['Duration (min)'] = (video_stats['End Time'] - video_stats['Start Time']) / 60
            
            st.dataframe(
                video_stats.style.format({
                    'Start Time': '{:.2f}',
                    'End Time': '{:.2f}',
                    'Duration (min)': '{:.2f}'
                }),
                use_container_width=True
            )
            
            # Visualizations
            st.markdown("#### Content Distribution")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("**Chunks per Video**")
                chunk_counts = video_stats.set_index('Video Number')['Chunk Count']
                st.bar_chart(chunk_counts)
            
            with chart_col2:
                st.markdown("**Duration per Video**")
                durations = video_stats.set_index('Video Number')['Duration (min)']
                st.bar_chart(durations)
    
    with explorer_tab2:
        st.markdown("#### Browse Content by Video")
        
        if 'number' in df.columns:
            # Get unique video numbers and their titles
            video_info = df.groupby('number')['title'].first().reset_index()
            video_options = [f"Video {row['number']}: {row['title']}" for _, row in video_info.iterrows()]
            
            selected_video_display = st.selectbox(
                "Select a video to browse:",
                options=video_options
            )
            
            if selected_video_display:
                # Extract video number from selection
                selected_video_num = selected_video_display.split(":")[0].replace("Video ", "").strip()
                video_chunks = df[df['number'] == selected_video_num].sort_values('start')
                
                st.markdown(f"**Total Chunks:** {len(video_chunks)}")
                st.markdown(f"**Video Number:** {selected_video_num}")
                st.markdown(f"**Video Title:** {video_chunks['title'].iloc[0] if len(video_chunks) > 0 else 'N/A'}")
                
                for idx, chunk in video_chunks.iterrows():
                    with st.expander(f"Chunk at {format_timestamp(chunk['start'])} - {format_timestamp(chunk['end'])}"):
                        st.write(chunk['text'])
    
    with explorer_tab3:
        st.markdown("#### Raw Database View")
        
        st.markdown("**Filter Options:**")
        
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            show_columns = st.multiselect(
                "Select columns to display:",
                options=['title', 'number', 'start', 'end', 'text', 'similarity'] if st.session_state.last_results is not None else ['title', 'number', 'start', 'end', 'text'],
                default=['title', 'number', 'start', 'end']
            )
        
        with filter_col2:
            max_rows = st.number_input("Max rows to display:", min_value=5, max_value=100, value=20)
        
        if show_columns:
            display_df = df[show_columns].head(max_rows)
            st.dataframe(display_df, use_container_width=True)
            
            # Download full dataset
            full_csv = df[show_columns].to_csv(index=False)
            st.download_button(
                label="Download Full Dataset (CSV)",
                data=full_csv,
                file_name=f"full_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Help Section
    st.markdown("---")
    with st.expander("HELP & DOCUMENTATION"):
        st.markdown("""
        ### How to Use This System
        
        #### Basic Search
        1. Enter your question in the search box
        2. Click "Search" button
        3. Review AI-generated response and relevant chunks
        
        #### Advanced Search
        1. Go to "Advanced Filters" tab
        2. Select specific videos or video numbers
        3. Enter your question
        4. Click "Search with Filters"
        
        #### Batch Processing
        1. Go to "Batch Query" tab
        2. Enter multiple questions (one per line)
        3. Click "Process Batch"
        
        ### Understanding Results
        
        - **Similarity Score**: Higher scores (closer to 1.0) indicate better matches
        - **Merged Chunks**: Each result contains ~5 consecutive subtitle segments
        - **Timestamps**: Shows exact location in video (MM:SS format)
        
        ### Tips for Better Results
        
        - Be specific in your questions
        - Use keywords from the course content
        - Try different phrasings if results aren't relevant
        - Adjust similarity threshold for more/fewer results
        
        ### System Requirements
        
        - Ollama running locally (for embeddings)
        - Groq API key configured (for AI responses)
        - embeddings.joblib file in project directory
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h4>RAG BASED AI TEACHING ASSISTANT</h4>
        <p>AI-Powered Intelligent Video Content Search System</p>
        <p>Built with Streamlit | Enhanced with Merged Chunks Technology</p>
        <p style="font-size: 0.8rem;">Developed by Suhaas S | IIIT Dharwad</p>
    </div>
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()