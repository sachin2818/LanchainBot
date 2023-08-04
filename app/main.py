
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("../.env")
    

from flask import Flask, request, render_template

import os

from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Weaviate
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate

from constants import PROMPT, COLLECTION_NAME


loader = PyPDFLoader("vector_source.pdf")
text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
document = loader.load()
docs = text_splitter.split_documents(document)
embeddings = OpenAIEmbeddings()
vectorstore = Weaviate.from_documents(
    docs,
    embeddings,
    weaviate_url=os.environ['WEAVIATE_URL'],
    by_text=False
)


prompt_template = PromptTemplate(
    template=PROMPT,
    input_variables=["context", "question"]
)


app=Flask(__name__)


@app.route("/query", methods=["POST"])
def query():
    memory = ConversationBufferMemory(memory_key="chat_history")
    args = (OpenAI(temperature=0.5), vectorstore.as_retriever())
    kwargs = {
        "combine_docs_chain_kwargs": {'prompt': prompt_template},
        "memory": memory
    }
    question_answer = ConversationalRetrievalChain.from_llm(*args, **kwargs)
    question = {"question": request.form["query"]}
    result = question_answer(question)
    return {"answer": result["answer"]}


@app.route("/chatbot",methods=["GET"])
def chatbot():
    return render_template("index.html")

def fetch_app():
    return app


if __name__ == '__main__':
    app.run(debug=True)

