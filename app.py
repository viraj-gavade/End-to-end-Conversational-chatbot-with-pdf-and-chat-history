
## Importing the required libraries
import streamlit as st 
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq.chat_models import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
import re
import tempfile

#Setting up the environment variables
import os 
from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
grok_api_key = os.getenv('GROQ_API_KEY')


embeddings = HuggingFaceEmbeddings( model_name = 'sentence-transformers/all-mpnet-base-v2')

st.title('Coversation rag with pdf uploads and chat history')
st.write('Upload pdfs and chat with thier content')

llm = ChatGroq(
    model='gemma2-9b-it',
    api_key=grok_api_key
)


session_id = st.text_input('Session ID :- ' , value='default_session')

if 'store' not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader('Choose a pdf file',type='pdf',accept_multiple_files=False)

## Processing uploaded files
if uploaded_files:
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(tmp_file.name)
            docs = loader.load()
            all_docs.extend(docs)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
    splits = text_splitter.split_documents(all_docs)
    vectorstore = Chroma.from_documents(splits,embeddings)
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = (
    "You are an AI assistant. Given the chat history and the latest user question, "
    "reformulate the user's question into a clear, self-contained standalone question. "
    "Do not answer the question; only provide the reformulated version.otherwise return as it is ")

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ('system',contextualize_q_system_prompt),
            ChatMessageHistory('chat_history'),
            ('human','{input}')
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

    #Answer question prompt 

    system_prompt = (''' You are an assistant for a question-answering task.
                Use the following retrieved context to answer the question.
                If you do not know the answer, say that you don't know.
                Keep your answer concise, using a maximum of three sentences. {context}''')
    

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ('system',system_prompt),
            MessagesPlaceholder('chat_history')
            ('human','{input}')
        ]
    )

    question_ans_chain = create_stuff_documents_chain(llm,qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever,question_ans_chain)


    def get_session_history(session_id)->BaseChatMessageHistory:
        if 'session_id' not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    

    coversational_rag_chain = RunnableWithMessageHistory(rag_chain,get_session_history,
                                                         input_messages_key='input',
                                                         history_messages_key='chat_history',
                                                         output_messages_key='answer')
    user_input = st.text_input('Your question:')
    if user_input:
        session_history = get_session_history(session_id)
        response = coversational_rag_chain.invoke(
            {'input':user_input},
            config={
                'configurable':{'session_id':session_id}
            },
        )

    st.write(st.session_state.store)
    st.write('Assistant: ' , response['answer'])
    st.write('Chat History: ' , session_history.messages)






