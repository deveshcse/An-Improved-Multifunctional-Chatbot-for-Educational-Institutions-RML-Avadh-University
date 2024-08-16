import re
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

#ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_fn = HuggingFaceEmbeddings(model_name=model_name)

api_key = "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"

#Main model client
chat = ChatGroq(temperature = 0,
                groq_api_key = api_key,
                model_name = "mixtral-8x7b-32768")

def document_loader():
    filename = "Dataset/RmluBOT.txt"
    with open(filename, errors="ignore") as file:
        contents = file.read()
    return contents


#Spliting the text by characters
print('spliting the text by Char and tokens...')
char_splitter = RecursiveCharacterTextSplitter(
    separators = ['\n', '\n\n', ' ', '. ', ', ', ''],
    chunk_size = 1000,
    chunk_overlap = 0.2
)

content = document_loader()
texts = char_splitter.create_documents([content])



vector_path = 'Vector_db/rmlu_Vector_db'
index_pkl = os.path.join(vector_path, 'index.pkl')
index_faiss = os.path.join(vector_path, 'index.faiss')

# Check if either file is missing
if not (os.path.exists(index_pkl) and os.path.exists(index_faiss)):
    vector_db = FAISS.from_documents(documents=texts, embedding=embedding_fn)
    vector_db.save_local(vector_path)

# Load the existing vector database
loaded_db = FAISS.load_local(
    vector_path,
    embeddings=embedding_fn,
    allow_dangerous_deserialization=True
)


# Create memory
memory = ConversationBufferWindowMemory(
    k=2,
    memory_key="chat_history",
    output_key="answer",
    return_messages=True,
)

# Create the ConversationalRetrievalChain
qa_conversation = ConversationalRetrievalChain.from_llm(
    llm=chat,
    chain_type="stuff",
    retriever=loaded_db.as_retriever(),
    return_source_documents=True,
    memory=memory
)

def RAG_Chain(query):
    print("Generating the response...")
    # Pass the query with the key 'question'
    response = qa_conversation({"question": query})
    return response.get("answer")

# while(1):
#     query = input("Enter the query : ")
#     if query == "1":
#         print("Thank You")
#         break
#     response = RAG_Chain(query)
#     print(response)
#     print()
#     print("Hope it is helpful for you...")
#     print("Enter 1 if you don't want to continue or ask the queries.")
#     print()