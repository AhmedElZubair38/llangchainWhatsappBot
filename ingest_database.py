from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"/mnt/c/Users/USER/crewai-ollama/data"
CHROMA_PATH = r"chroma_db"

# initiate the embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# initiate the vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# loading the PDF document
loader = PyPDFDirectoryLoader(DATA_PATH)

raw_documents = loader.load()

# splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

# creating the chunks
chunks = text_splitter.split_documents(raw_documents)

# creating unique ID's
uuids = [str(uuid4()) for _ in range(len(chunks))]

# adding chunks to vector store
vector_store.add_documents(documents=chunks, ids=uuids)

# vector_store.add_documents(documents=chunks, ids=uuids)



# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_chroma import Chroma
# from uuid import uuid4
# from dotenv import load_dotenv
# from chromadb.config import Settings
# import os
# import pathlib

# from dotenv import load_dotenv
# load_dotenv()

# # Configuration - use pathlib for cross-platform paths
# DATA_PATH = pathlib.Path(r"/mnt/c/Users/USER/llangchainwhatsappbot/data")
# CHROMA_PATH = pathlib.Path("chroma_db")
# os.makedirs(CHROMA_PATH, exist_ok=True)

# # Initialize embeddings
# embeddings_model = OpenAIEmbeddings(
#     model="text-embedding-3-large"
# )

# # Correct Chroma configuration
# vector_store = Chroma(
#     collection_name="aquasprint_knowledge",
#     embedding_function=embeddings_model,
#     persist_directory=str(CHROMA_PATH)
# )

# # Rest of the script remains the same...
# try:
#     # Load PDF document with error handling
#     print(f"Loading documents from {DATA_PATH}")
#     loader = PyPDFDirectoryLoader(
#         str(DATA_PATH),
#         recursive=False,
#         silent_errors=False
#     )
#     raw_documents = loader.load()
#     print(f"Successfully loaded {len(raw_documents)} document(s)")
    
#     if not raw_documents:
#         raise ValueError("No documents found in the specified path")

# except Exception as e:
#     print(f"Document loading failed: {str(e)}")
#     exit()

# # Configure text splitter with proper regex handling
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len,
#     separators=[
#         "\n\n", 
#         "\n", 
#         r"(?<=\. )",  # Raw string for regex
#         " ", 
#         ""
#     ],
#     is_separator_regex=True,
#     keep_separator=True
# )

# # Process documents
# chunks = text_splitter.split_documents(raw_documents)
# print(f"Split into {len(chunks)} chunks")

# # Add metadata to chunks
# for chunk in chunks:
#     chunk.metadata.update({
#         "source": "aquasprint_handbook",
#         "content_type": "swimming_program",
#         "processed": "true"
#     })

# # Store vectors with verification
# try:
#     print("Storing vectors in ChromaDB...")
#     ids = [str(uuid4()) for _ in chunks]
#     vector_store.add_documents(chunks, ids=ids)
    
#     # Verification check
#     collection_info = vector_store.get()
#     stored_count = len(collection_info['ids'])
#     if stored_count != len(ids):
#         print(f"Warning: Expected {len(ids)} vectors, stored {stored_count}")
#     else:
#         print(f"Successfully stored {stored_count} vectors")
        
#     # Persist to disk
#     vector_store.persist()
#     print(f"Database persisted to {CHROMA_PATH}")

# except Exception as e:
#     print(f"Vector storage failed: {str(e)}")
#     exit()

# # Diagnostic output
# print("\nFirst 3 chunks preview:")
# for i, chunk in enumerate(chunks[:3]):
#     content_preview = chunk.page_content[:100].replace('\n', ' ')
#     print(f"Chunk {i+1}: {content_preview}...")