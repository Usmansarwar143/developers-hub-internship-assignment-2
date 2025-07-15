import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

# Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Streamlit UI setup
st.set_page_config(page_title="Context-Aware Chatbot", layout="wide")
st.title("🤖 Context-Aware Chatbot with Hugging Face")

@st.cache_resource
def load_vectorstore():
    # Load document
    loader = TextLoader("data/knowledge.txt")
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Load vector store
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

# Memory for context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# LLM setup

# Load tokenizer and model
model_name = "tiiuae/falcon-rw-1b" # You can try smaller models if this is too heavy
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    offload_folder="./offload"  # <- 👈 this tells Transformers where to offload parts of the model
)

# Create a text generation pipeline
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.5,
    do_sample=True,
    top_k=50,
    top_p=0.95
)

# Use it in LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)



# Create the RAG chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.text_input("Ask something:")

if user_question:
    response = qa_chain.run(user_question)
    st.session_state.chat_history.append(("You", user_question))
    st.session_state.chat_history.append(("Bot", response))

# Display conversation
for sender, msg in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {msg}")
