import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder


load_dotenv()
#loading environment variables
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")


st.title("Coversational RAG with PDF Upload and chat History")
st.write("Upload one or more PDF and ask questions about its content")

#input to the page

groq_api_key = st.text_input("Enter GROQ API Key",type="password")

if groq_api_key:
    llm = ChatGroq(model_name="Gemma2-9b-It", groq_api_key=groq_api_key)
    
    session_id = st.text_input("Session ID",value = "Default_Session")

    if "store" not in st.session_state:
        st.session_state.store={}
    
    uploaded_files = st.file_uploader("Uploader the PDF files here",type="pdf",accept_multiple_files=True)
    documents = []
    if uploaded_files:
        documents = []  
        for uploaded_file in uploaded_files:
            temp_pdf = './temp.pdf'
            with open(temp_pdf,'wb') as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)
            document_content = loader.load()
            documents.extend(document_content)
    
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
        splitted_docs = text_splitter.split_documents(documents)
        embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")    
        vector_store = FAISS.from_documents(splitted_docs,embeddings)
        retriever = vector_store.as_retriever()
    
        system_summaried_question = """ Given a chat history and latest user question form a standalone question which
         represent the context of the chat history. Do not answer the question. just form a question.
         just reframe the question if it is neded or else return it as it is
          """
        system_summaried_prompt = ChatPromptTemplate.from_messages(
            [ 
               ("system",system_summaried_question),
               MessagesPlaceholder("chat_history"),
               ("human","{input}")
            ]
        )

        history_retriever = create_history_aware_retriever(llm,retriever,system_summaried_prompt)

        system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
    
        qa_prompt = ChatPromptTemplate.from_messages(
           [
              ("system",system_prompt),
               MessagesPlaceholder("chat_history"),
              ("human","{input}")
            ]
        )

        llm_prompt_chain = create_stuff_documents_chain(llm,qa_prompt)
        retrieval_chain = create_retrieval_chain(history_retriever,llm_prompt_chain)

        def get_session_history(session:str)->BaseChatMessageHistory: 
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        coversational_rag_chain = RunnableWithMessageHistory(retrieval_chain,
                                        get_session_history,
                                        input_messages_key = "input",
                                        history_messages_key = "chat_history",
                                        output_messages_key = "answer")
     
        user_input = st.text_input("Please Ask Your Question Here")
        if user_input:
           session_history = get_session_history(session_id)
           response = coversational_rag_chain.invoke({"input":user_input},
                                              {"configurable":{"session_id":session_id}},
                                            )
           st.write("Assitant:",response['answer']) 
           st.write("Session store:",st.session_state.store)   
           st.write("Session History:",session_history.messages)
        st.write("Session store:",st.session_state.store)  
