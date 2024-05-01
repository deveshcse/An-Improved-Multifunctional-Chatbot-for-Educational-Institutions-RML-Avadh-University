from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI


print("initializing embeddings..and llm.........")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# llm = Ollama(model="mistral")
llm = ChatGoogleGenerativeAI(google_api_key="AIzaSyD9MBZz-xHfGuZcWbw-YYVWCRNLdfu-jxk",
                             model="gemini-pro",
                             temperature=0.7,
                             top_p=0.85)


def rag_with_llm(retriever, llm):
    print("creating retrieval qa chain object........ ")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return chain


def document_loader():
    filename = "RmluBOT.txt"
    with open(filename, errors="ignore") as file:
        contents = file.read()
        # print(contents)
    return contents


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)


content = document_loader()
text = text_splitter.create_documents([content])
vector_db = FAISS.from_documents(documents=text, embedding=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 20})
qa_chain = rag_with_llm(retriever=retriever, llm=llm)





