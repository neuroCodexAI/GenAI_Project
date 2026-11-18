import os
import uuid
import shutil
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

UPLOAD_DIR = "./doc_files/"

# ---- session state ----
defaults = {
    "document_uploaded": False,
    "agent": None,
    "messages": [],
    "thread_id": str(uuid.uuid4()),
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def process_document(path):
    docs = PyPDFDirectoryLoader(path).load()
    docs = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)

    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    vector_db = InMemoryVectorStore.from_documents(documents=docs, embedding=embeddings)

    llm = ChatGroq(model="openai/gpt-oss-20b")

    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve documents relevant to a query from the knowledge base."""
        results = vector_db.similarity_search(query=query, k=3)
        return "\n\n".join(doc.page_content for doc in results)

    system_prompt = (
        "You are a helpful assistant that answers questions using retrieved context. "
        "The knowledge base consists of the uploaded documents. "
        "Always use the `retrieve_context` tool for questions requiring external knowledge."
    )

    st.session_state.agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )
    st.session_state.document_uploaded = True


# ---- upload UI ----
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader("Select PDF File", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing..."):
            shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            for file in uploaded:
                with open(os.path.join(UPLOAD_DIR, file.name), "wb") as f:
                    f.write(file.getvalue())
            process_document(UPLOAD_DIR)
        st.rerun()

# ---- chat UI ----
if st.session_state.document_uploaded and st.session_state.agent:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    query = st.chat_input("Ask anything related to uploaded documents....")
    if query:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        response = st.session_state.agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": st.session_state.thread_id}},
        )
        answer = response["messages"][-1].content

        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})