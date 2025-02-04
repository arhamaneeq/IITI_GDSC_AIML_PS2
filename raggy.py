import streamlit as st
import tempfile

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

st.title("R A G G Y 🤖")
st.header("It'll do your thinking for you. 🧠🤯")

# Upload Document from User
uploaded_file = st.file_uploader("Upload the Context File Here")

# Temporary variable to store the vector store
vectorstore = None

# Function to handle the document processing and query response
def process_document_and_query(uploaded_file, query):
    global vectorstore

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())  # Save the uploaded file's content to temp file
            temp_file_path = temp_file.name  # Get the path of the temp file

        # Load Document
        documents = PyPDFLoader(temp_file_path).load()

        # Split it into Chunks
        chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"}
        )

        # Create vector store from documents and embeddings
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Save the vector store locally
        vectorstore.save_local("faiss_index_")

    # Load the vector store (if it exists) and use it for retriever-based QA
    if vectorstore:
        persisted_vectorstore = FAISS.load_local("faiss_index_", embeddings, allow_dangerous_deserialization=True)
        retriever = persisted_vectorstore.as_retriever()

        # Initialize the language model (Ollama)
        llm = Ollama(model="llama3.1")

        # Construct a retrieval-based chain and get the response
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Run the query through the chain
        response = qa_chain.run(query)
        return response
    return "Please upload a file and try again."

# Get User Input for Query
prompt = st.chat_input("Ask Raggy anything!")

if prompt:
    response = process_document_and_query(uploaded_file, prompt)
    st.write(response)



