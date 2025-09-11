import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings,ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time

from dotenv import load_dotenv
load_dotenv()

## Load the NVIDIA API Key
nvidia_api_key=st.secrets["api_keys"]["nvidia"]
os.environ["NVIDIA_API_KEY"]=nvidia_api_key

def vector_embeddings():

    if "vectors" not in st.session_state:
        st.session_state.embeddings=NVIDIAEmbeddings() #Embedding setup
        st.session_state.loader=PyPDFDirectoryLoader("./investment_approach") #Data ingestion
        st.session_state.docs=st.session_state.loader.load() #Document loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50) #Chunk creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30]) #Splitting the doc
        print("Hello")
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #Vector store creation

st.set_page_config(
    page_title="Investment Wizard using NVidia NIM",
    page_icon="📈💰🏦💹"
)

st.title("Investment Wizard using NVidia NIM 💰")
llm=ChatNVIDIA(model="meta/llama-3.1-8b-instruct")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)

prompt1=st.text_input("Enter your question regrding investment in India")

if st.button("Documents Embedding"):
    vector_embeddings()
    st.write("Vector store DB is ready")

import time

if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time:",time.process_time()-start)
    st.write(response['answer'])

    #With streamlit expander
    with st.expander("Document Similarity Search"):
        #Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------")