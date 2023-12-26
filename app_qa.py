import os
from typing import Optional
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import CTransformers
from langchain.chat_models import ChatOpenAI

import chromadb
from chromadb.utils import embedding_functions


import chainlit as cl

# Load environment variables
load_dotenv()

# Define environment variables
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_KEY")
CHAINLIT_API_KEY = os.getenv("CHAINLIT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHROMA_DATA_PATH = r"./RAG_database"
DATA_DIR = r"./data"
COLLECTION_NAME = "LLM-Test-Instadeep-RAG"
EMBED_MODEL = "allenai/scibert_scivocab_uncased"


EMBED_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(
     model_name=EMBED_MODEL
 )

client = chromadb.PersistentClient(path = CHROMA_DATA_PATH)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=EMBED_FUNCTION)

huggingface_embeddings = HuggingFaceEmbeddings(
                model_name="allenai/scibert_scivocab_uncased",
                model_kwargs={"device": "cpu"},
            )

langchain_chroma = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=huggingface_embeddings,
)

# Initialize the language model
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", streaming=True)

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.AppUser]:
  # Fetch the user matching username from your database
  # and compare the hashed password with the value stored in the database
  if (username, password) == ("admin", "admin"):
    return cl.AppUser(username="admin", role="ADMIN", provider="credentials")
  else:
    return None

@cl.on_chat_start
async def main():
    qa_chain = RetrievalQA.from_chain_type(llm = llm,
                                    chain_type="stuff",
                                    return_source_documents = True,
                                    verbose=False,                       
                                    retriever = langchain_chroma.as_retriever(search_kwargs={"k": 1}))
    cl.user_session.set("retrievalQA_chain", qa_chain)

@cl.on_message
async def main(message:str):

    retrievalQA_chain = cl.user_session.get("retrievalQA_chain")
    callback_handler = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True)
    callback_handler.answer_reached = True
    results = await retrievalQA_chain.acall(str(message.content), callbacks=[callback_handler])
    print(results)