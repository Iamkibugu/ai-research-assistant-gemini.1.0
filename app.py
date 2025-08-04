import os
import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Set Google Gemini API key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Streamlit UI
st.set_page_config(page_title="Gemini AI Research Assistant", layout="wide")
st.title("📄 AI Research Assistant (Gemini Powered)")
st.write("Upload a PDF and ask questions about it.")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
user_question = st.text_input("Ask a question about your PDF")

if uploaded_file and user_question:
    # Extract text
    reader = PyPDF2.PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(text)

    # Generate embeddings & create vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_texts(texts, embedding=embeddings, persist_directory="./chroma_db")

    # RAG: Retrieval + Gemini Pro Answer
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    with st.spinner("Searching your document..."):
        response = qa.run(user_question)

    st.success("Answer:")
    st.write(response)
