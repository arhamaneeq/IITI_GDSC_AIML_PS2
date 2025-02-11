from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from llms_chain import load_vectordb, create_embeddings
import pypdfium2

def get_pdf_text(pdfs_bytes):
    return [extract_text_from_pdfs(pdf_bytes) for pdf_bytes in pdfs_bytes]

def extract_text_from_pdfs(pdf_bytes):
    pdf_file = pypdfium2.PdfDocument(pdf_bytes)
        
    return "\n".join(pdf_file.get_page(page_number).get_textpage().get_text_range() for page_number in range(len(pdf_file)))
    
def get_text_chunks(text):
    Splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 100, separators=["\n", "\n\n"])
    return Splitter.split_text(text)

def get_document_chunks(text_list):
    documents = []
    for text in text_list:
        for chunk in get_text_chunks(text):
            documents.append(Document(page_content=chunk))

    return documents

def add_docs_to_db(pdfs_bytes):
    texts = get_pdf_text(pdfs_bytes)                # extract text from pdfs
    documents = get_document_chunks(texts)          # split text into chunks and create documents
    vector_db = load_vectordb(create_embeddings())  # load vector db
    vector_db.add_documents(documents)              # add documents to vector db



