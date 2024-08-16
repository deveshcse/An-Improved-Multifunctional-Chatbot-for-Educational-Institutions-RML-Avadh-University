import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter


import getpass
import os

os.environ["GROQ_API_KEY"] = getpass.getpass("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")

# Initialize embeddings and llm
print("Initializing embeddings and LLM...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


from langchain_groq import ChatGroq

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    
)
if llm:
    print(llm)

llm = Ollama(model="mistral")

# Define function to create QA chain
def rag_with_llm(retriever, llm):
    print("Creating retrieval QA chain object...")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return chain

# Define function to load document
def document_loader():
    filename = "Dataset/RmluBOT.txt"
    with open(filename, errors="ignore") as file:
        contents = file.read()
    return contents

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

# Load document and create text chunks
content = document_loader()
text = text_splitter.create_documents([content])
if text:
    print(text)

# Create vector database from documents
vector_db = FAISS.from_documents(documents=text, embedding=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
if retriever:
    print(retriever)
# Create QA chain
qa_chain = rag_with_llm(retriever=retriever, llm=llm)
if qa_chain:
    print(qa_chain)
# Streamlit UI
st.title("An Improved Multifunctional Chatbot for Educational Institutions RML Avadh University")

# Input text box for user query
query = st.text_area("Enter your question:")

# Button to generate response
if st.button("Generate Response"):
    # Pass query to QA chain and get response
    response = qa_chain(query)
    # Display response
    st.write("Response:", response['result'])
