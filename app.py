import streamlit as st
import google.generativeai as genai
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import sys

st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🏥]",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def initialize_rag_system():
    try:
        from medical_rag_system import MedicalRAGSystem
        rag_system = MedicalRAGSystem()
        return rag_system, None
    except Exception as e:
        return None, f"Error initializing RAG system: {str(e)}"

def generate_medical_answer(query, context_chunks, api_key):
    if not context_chunks:
        return "I couldn't find relevant medical information to answer this question in the available records."

    context_text = "\n\n".join([
        f"--- MEDICAL NOTE {i+1} (Specialty: {chunk['metadata']['medical_specialty']}) ---\n{chunk['content']}"
        for i, chunk in enumerate(context_chunks)
    ])

    prompt = f"""You are a medical research assistant. Answer the question based ONLY on the provided medical context from clinical notes.

MEDICAL CONTEXT:
{context_text}

QUESTION: {query}

IMPORTANT INSTRUCTIONS:
- Answer using ONLY the information from the medical context above
- If the context doesn't contain relevant information, say "I cannot find specific information about this in the available medical records"
- Be precise and medically accurate
- Do not make up or hallucinate information
- Mention which medical specialty the information comes from when relevant
- Keep answers concise but informative

ANSWER:"""

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating answer: {str(e)}"

st.markdown('<div class="main-header">🏥 Medical RAG Assistant</div>', unsafe_allow_html=True)
st.markdown("**Ask medical questions based on 3,898 clinical transcriptions across 39 medical specialties**")

with st.sidebar:
    st.header(" Configuration")

    api_key = st.text_input(
        "Google AI Studio API Key",
        type="password",
        help="Get free API key from https://aistudio.google.com/"
    )

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.write("**How to get API Key:**")
    st.write("1. Go to [Google AI Studio](https://aistudio.google.com/)")
    st.write("2. Sign in with Google account")
    st.write("3. Click 'Get API Key' and create new key")
    st.write("4. Paste the key here")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button(" Initialize Medical RAG System", use_container_width=True):
        if not api_key:
            st.error("Please enter your Google AI Studio API key first")
        else:
            with st.spinner("Loading medical database..."):
                rag_system, error = initialize_rag_system()
                if rag_system:
                    st.session_state.rag_system = rag_system
                    st.session_state.api_key = api_key
                    st.success(" Medical RAG System Ready!")
                    st.write(f"• Medical chunks: {len(rag_system.chunks):,}")
                    st.write(f"• Specialties: {len(set(m['medical_specialty'] for m in rag_system.metadata))}")
                    st.write(f"• Vector dimension: {rag_system.index.d}")
                else:
                    st.error(f" {error}")

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'history' not in st.session_state:
    st.session_state.history = []

if st.session_state.rag_system:
    st.header(" Medical Question & Answer")

    query = st.text_input(
        "Ask your medical question:",
        placeholder="e.g., What are common treatments for allergies? What symptoms indicate asthma?",
        key="query_input"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        num_chunks = st.slider("Sources to retrieve", 1, 5, 3)

    if query and st.session_state.get('api_key'):
        with st.spinner(" Searching medical database..."):
            chunks = st.session_state.rag_system.retrieve_similar_chunks(query, k=num_chunks)
            answer = generate_medical_answer(query, chunks, st.session_state.api_key)

            st.session_state.history.append({
                'query': query,
                'answer': answer,
                'chunks_used': len(chunks),
                'timestamp': pd.Timestamp.now()
            })

        st.subheader(" Answer:")
        st.write(answer)

        with st.expander(f" View Source Documents ({len(chunks)} found)"):
            for i, chunk in enumerate(chunks):
                st.markdown('<div class="source-box">', unsafe_allow_html=True)
                st.write(f"**Source {i+1}** | **Specialty:** {chunk['metadata']['medical_specialty']} | **Similarity Score:** {chunk['similarity_score']:.3f}")
                st.write(f"**Content:** {chunk['content'][:400]}...")
                st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.subheader(" Recent Questions")
        for i, item in enumerate(reversed(st.session_state.history[-3:])):
            st.write(f"**Q:** {item['query']}")
            st.write(f"**A:** {item['answer'][:200]}...")
            st.write(f"*Sources used: {item['chunks_used']}*")
            st.divider()

else:
    st.info(" Welcome! Please enter your Google AI Studio API key and initialize the system in the sidebar to start asking medical questions.")

with st.expander("ℹ System Information"):
    st.write("""
    **Medical RAG System Overview:**

    - **Data Source:** 3,898 clinical medical transcription records
    - **Medical Content:** 29,713 processed text chunks
    - **Specialties Covered:** 39 different medical specialties
    - **Search Technology:** FAISS vector similarity search
    - **AI Model:** Google Gemini for answer generation
    - **Key Feature:** Provides source citations for transparency

    **How it works:**
    1. Your question is converted to a vector embedding
    2. System finds the most similar medical text chunks
    3. Gemini generates an answer using only the retrieved context
    4. Sources are provided for verification

    **Note:** This system provides information from medical records but is not a substitute for professional medical advice.
    """)

st.markdown("---")
st.markdown("*Built with Streamlit, FAISS, and Google Gemini • Medical RAG System*")
