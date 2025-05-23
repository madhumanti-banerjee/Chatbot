import streamlit as st
import openai
import os
import pickle
import sqlite3
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import time

# Streamlit config
st.set_page_config(page_title="PDF FAQ Chatbot", layout="wide")
st.title("📄 PDF FAQ Chatbot with OpenAI")

# Set OpenAI API key (manually or via sidebar input for security)
api_key = st.secrets["OPENAI_API_KEY"]
if not api_key:
    st.warning("Please enter your OpenAI API key.")
    st.stop()
openai.api_key = api_key

# File path to your local PDF
PDF_PATH = "faq.pdf"

# Session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# Process the PDF
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma  # ✅ using Chroma instead of FAISS

def process_pdf(file_path, chunk_size, chunk_overlap):
    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma.from_documents(texts, embedding=embeddings)  # ✅ uses `texts` here

    return vectorstore

# Only process once
if st.session_state.vectorstore is None:
    st.info("📄 Processing PDF, please wait...")
    try:
        st.session_state.vectorstore = process_pdf(PDF_PATH, 500, 50)
        st.success("✅ PDF processed successfully.")
    except Exception as e:
        st.error(f"Failed to process PDF: {e}")
        st.stop()

# Chat UI
st.subheader("💬 Ask a question about the PDF")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question about the PDF...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        start = time.time()
        docs = st.session_state.vectorstore.similarity_search(user_input, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""You are a helpful assistant. Answer the question based only on the context below.

Context:
{context}

Question: {user_input}
"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response['choices'][0]['message']['content']
        except Exception as e:
            answer = f"❌ Error: {e}"

        elapsed = time.time() - start
        final_response = f"{answer}\n\n*Response time: {elapsed:.2f} sec*"
        st.chat_message("assistant").markdown(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
