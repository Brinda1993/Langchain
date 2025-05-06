import streamlit as st
import os
from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun,DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain.agents import initialize_agent,AgentType
from langchain_groq import ChatGroq
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper = wiki_wrapper)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300)
arxiv_tool = ArxivQueryRun(api_wrapper = arxiv_wrapper)

search = DuckDuckGoSearchResults(name="Search for latest information")

st.title("Langchain Search with Agents and tools")

st.sidebar.title("Settings")

groq_api_key = st.sidebar.text_input("Enter the GROQ API Key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role":"AI Assitant","content":"Hi I am a Chatbot. How Can I assist You?"
    }]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:=st.chat_input("Ask me the Question..."):
    st.session_state.messages.append({"role":"User","content":prompt})
    st.chat_message("User").write(prompt)
    
    llm = ChatGroq(model_name="Llama3-8b-8192", groq_api_key=groq_api_key)
    tools = [search,wiki_tool,arxiv_tool]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)   
    
    with st.chat_message("AI Assitant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role":"AI Assitant","content":response})
        st.write(response)
        st.write(st.session_state.messages)