import os
from typing import Optional

# Import necessary libraries
from dotenv import load_dotenv
from io import BytesIO
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
import chromadb
import PyPDF2
import chainlit as cl

# Load environment variables
load_dotenv()

# Define environment variables
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_KEY")
CHAINLIT_API_KEY = os.getenv("CHAINLIT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define paths and collection name
CHROMA_DATA_PATH = r"./embeddings_database" 
COLLECTION_NAME = "LLM-Test-Instadeep"

# Initialize the language model
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")


def summarize(files):
    """Summarizes the text content of a PDF file."""

    uploaded_file = files[0]

    with BytesIO(uploaded_file.content) as pdf_stream:  # Use context manager
        pdf = PyPDF2.PdfReader(pdf_stream)

        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=30)
        texts = text_splitter.split_text(pdf_text)
        docs = [Document(page_content=t) for t in texts]

        chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False)
        summary = chain.run(docs)
        return summary


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.AppUser]:
    """Authenticates users based on credentials."""

    # Implement actual authentication logic here (e.g., database lookup)
    if (username, password) == ("admin", "admin"):
        return cl.AppUser(username="admin", role="ADMIN", provider="credentials")
    else:
        return None


@cl.on_chat_start
async def main():
    """Handles initial chat setup and PDF summarization."""

    client = chromadb.PersistentClient(CHROMA_DATA_PATH)
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name="allenai/scibert_scivocab_uncased",
        model_kwargs={"device": "cpu"},
    )

    langchain_chroma = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=huggingface_embeddings,
    )
    retriever = langchain_chroma.as_retriever(search_kwargs={"k": 3}, search_type="similarity")
    cl.user_session.set("retriever", retriever)

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF to summarize it",
            accept=["text/plain", "application/pdf"],
            max_size_mb=10,
        ).send()

    await cl.Message(content=f"Processing paper...").send()
    summary = await cl.make_async(summarize)(files)
    await cl.Message(content=summary).send()


@cl.on_message
async def main(message: str):
    """Recommends papers based on user input."""

    retriever = cl.user_session.get("retriever")
    docs = retriever.get_relevant_documents(message.content)
    titles = [docs[i].metadata["title"] for i in range(3)]
    response = ("I recommend to you the following 3 papers similar to your input message:\n\n\n1- {}\n\n2- {}\n\n3- {}\n\n").format(titles[0], titles[1], titles[2])
    msg = cl.Message(content=response)
    await msg.send()
  
