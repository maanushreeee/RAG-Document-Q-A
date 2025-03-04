import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Set API Key
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize LLM
llm = ChatGroq(api_key=groq_api_key, model_name='llama3-8b-8192')

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    '''
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    
    <context>
    {context}
    <context>
    
    Question: {input}
    '''
)

# Function to create vector embeddings
def create_vector_embeddings():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader('data')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Streamlit UI
st.title('Research Paper Q&A')

# User input
user_prompt = st.text_input('Enter your query from research paper')

# Button to initialize vector database
if st.button('Document Embedding'):
    create_vector_embeddings()
    st.write('Vector Database is ready')

# Ensure vectors are initialized before querying
if user_prompt:
    if 'vectors' not in st.session_state:
        st.warning("Please click 'Document Embedding' to initialize the vector database first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        st.write(f'Response Time: {time.process_time() - start:.2f} seconds')

        st.write(response['answer'])

        # Display similar documents
        with st.expander('Document Similarity Search'):
            for i, doc in enumerate(response.get('context', [])):
                st.write(doc.page_content)
