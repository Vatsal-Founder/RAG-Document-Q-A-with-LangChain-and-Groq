import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()



os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")    
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]= "RAG Document Q&A"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")

llm=ChatGroq(api_key=groq_api_key, model="llama3-70b-8192")

chat_prompt_template = PromptTemplate(
    input_variables=["context", "input"],
    template=
    """
    Hey, answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    Context: {context}
    Question: {input}
    """
)

def create_vector_embedding():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings()
        st.session_state.loader = PyPDFLoader("LLM Interview Questions.pdf")
        st.session_state.documents = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_docs=st.session_state.text_splitter.split_documents(st.session_state.documents[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)

st.title("RAG Document Q&A with LangChain and Groq")

if st.button("Create Vector Embeddings"):
    create_vector_embedding()
    st.write("Vector embeddings created successfully!")
    

user_question = st.text_input("Enter your question:")
import time
start_time = time.process_time()

if user_question:
    retriver=st.session_state.vectors.as_retriever()
    doc_chain= create_stuff_documents_chain(llm,chat_prompt_template)

    reg_chain=create_retrieval_chain(retriver, doc_chain)
    start_time = time.process_time()

    response = reg_chain.invoke({"input": user_question})

    response_time = time.process_time() - start_time
    print(f'Response time is {response_time}')

    st.write(response['answer'])

    with st.expander("Show Context"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("---")
