import streamlit as st
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
)  # 이건 더이상 사용하지 말라고 하나 langchain-unstructured 충돌하고 있기 때문에 임시로 쓴다
from langchain_text_splitters import (
    CharacterTextSplitter,
)
from langchain_openai import OpenAIEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai.chat_models import ChatOpenAI

st.set_page_config(page_title="Document GPT", page_icon="📒")


class ChatCallBackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        # self.message = f"{self.message}{token}"
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallBackHandler(),
    ],
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_resource(show_spinner="Embedding file...")
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
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = Chroma.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_histry():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return ("/n/n".join(document.page_content for document in docs),)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


# 여기서 부터 st 형식
st.title("Document GPT")

st.markdown(
    """
    Welcom!
            
    Use this chatbot to as questions to an AI about your files

"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf .docx file", type=["pdf", "txt", "docx"]
    )

# select = st.selectbox("", ["일반", "깊게 생각하기"])

if file:
    retriever = embed_file(file)

    send_message("I'm ready! Ask away!", "AI", save=False)
    paint_histry()
    message = st.chat_input("Ask Anything..")
    if message:
        send_message(message, "human")

        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)

else:
    st.session_state["messages"] = []
