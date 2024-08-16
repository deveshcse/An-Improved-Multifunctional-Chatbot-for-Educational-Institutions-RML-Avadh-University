from flask import Flask, request, jsonify

app = Flask(__name__)

# Paste your initialization code here
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter

print("initializing embeddings..and llm.........")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = Ollama(model="mistral")


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
    filename = "Dataset/RmluBOT.txt"
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


@app.route('/api/answer', methods=['POST'])
def get_answer():
    data = request.json
    question = data['question']

    # Perform inference
    answer = qa_chain.predict(question)

    return jsonify({'answer': answer})


if __name__ == '__main__':
    app.run(debug=True)
