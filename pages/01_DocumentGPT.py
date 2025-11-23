import time
import streamlit as st
import dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
)  # 이건 더이상 사용하지 말라고 하나 langchain-unstructured 충돌하고 있기 때문에 임시로 쓴다
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_openai import OpenAIEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

dotenv.load_dotenv()

st.set_page_config(page_title="Document GPT", page_icon="📒")


def embed_file(file):
    file_content = file.read()
    file_path = f"./.caches/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.caches/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600,
        chunk_overlap=50,
    )
    loader = UnstructuredFileLoader("./files/chapter_one.txt")
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = Chroma.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


st.title("Document GPT")

st.markdown(
    """
    Welcom!
            
    Use this chatbot to as questions to an AI about your files

"""
)

file = st.file_uploader("Upload a .txt .pdf .docx file", type=["pdf", "txt", "docx"])

if file:
    retriever = embed_file(file)
    retriever.inovoke("winston")
