from dotenv import load_dotenv
load_dotenv()



from langchain_community.document_loaders import PyPDFLoader , PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
import streamlit as st


### data in st session 

if "documents_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
    
if "agent" not in st.session_state:
    st.session_state.agent = None
    
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    
if "messages" not in st.session_state:
    st.session_state.messages = []


def process_document(path):
    
    ## load the document
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()


    ### Splitter docs 

    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    docs = splitter.split_documents(docs)

    ## Embeeding and Vector Store 
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    ### Create agent  - tool, llm , system prompt
    llm = ChatGroq(model = "openai/gpt-oss-20b")

    @tool
    def retrieve_context(query:str):
        """Retrieve documents relevant to a query from the knowledge base. """
        context = ""
        docs = vector_db.similarity_search(query=query, k = 3)
        for doc in docs:
            context = doc.page_content + "\n\n"
            
        return context 

    system_prompt = """you are a helpful assistant that answers questions using retrieved context. my knowledge base consists of the details from the uploaded document. always use the `retrieve_context` tool for questions requiring external knowledge."""

    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools =[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )
    
    st.session_state.agent = agent
    st.session_state.document_uploaded = True



### Upload UI 
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label = "Select PDF File", type = ["pdf"], accept_multiple_files= True)
    if  uploaded:
        with st.spinner("Processing..."):
            path = "./doc_files/"
            for file in uploaded:
                with open(path + file.name, 'wb') as f:
                    f.write(file.getvalue())

            process_document(path)
            st.rerun()
    

### Chat UI 

if st.session_state.document_uploaded and st.session_state.agent:
    query = st.chat_input("Ask anything related to uploaded documents....")
    if query:
        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke(
            {"messages":[{"role":"user","content":query}]},
            {"configurable":{"thread_id":1}}
        )
        
        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)






# while True:
#     query = input("User:")
#     if query.lower() == "quit":
#         break
#     response = agent.invoke({"messages":[{"role":"user", "content":query}]},
#                             {"configurable":{"thread_id":1}}
#                             )
#     result = response["messages"][-1].content
    
#     print("AI:", result)
