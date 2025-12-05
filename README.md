Medical RAG Assistant
A Retrieval-Augmented Generation (RAG) system for medical question answering, built with Streamlit, FAISS, and Google Gemini.

🏥 Features
29,713 medical text chunks from 3,898 clinical transcriptions
39 medical specialties covered
Semantic search using FAISS vector database
AI-powered answers using Google Gemini
Source citation for transparency
Web interface with Streamlit
Quick Start
Get API Key: Free from Google AI Studio
Enter API Key: In the app sidebar
Initialize System: Click "Initialize Medical RAG System"
Ask Questions: Type your medical questions
Project Structure
medical-rag-assistant/
├── app.py                          # Main Streamlit app
├── medical_rag_system.py           # RAG system module
├── requirements.txt                # Dependencies
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
└── medical_rag/
    └── vector_store/               # Vector database
        ├── medical_faiss.index
        └── vector_metadata.pkl
Medical Disclaimer
This system provides information from medical records for educational purposes only. It is not a substitute for professional medical advice.
