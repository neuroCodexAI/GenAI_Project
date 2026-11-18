from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

llm = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")

st.title("🤖 AskBuddy - AI Qna Bot")
st.markdown("My Qna Bot with Langchain and Google Gemini !")

if "messages" not in st.session_state:
    st.session_state.messages =[]
    
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)
    

query = st.chat_input("Ask anythings ?")
if query:
    st.session_state.messages.append({"role":"user", "content":query})
    st.chat_message("user").markdown(query)
    res = llm.invoke(query)
    st.chat_message("Ai").markdown(res.text())
    st.session_state.messages.append({"role":"ai", "content":query})


# while True:
#     query  = input("User:")
    
#     if query.lower() in ["exit", "quit", "bye"]:
#         break
    
#     res = llm.invoke(query)
#     print("AI:", res.content, '\n')
    
    

