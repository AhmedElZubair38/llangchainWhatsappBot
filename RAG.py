from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = r"/mnt/c/Users/USER/llangchainwhatsappbot/data"
CHROMA_PATH = r"chroma_db"

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

markdown_file = f"{DATA_PATH}/info.md"
loader = UnstructuredMarkdownLoader(file_path=markdown_file)
raw_documents = loader.load()

doc_ids = [str(uuid4()) for _ in range(len(raw_documents))]

vector_store.add_documents(documents=raw_documents, ids=doc_ids)

print(f"Successfully stored {len(raw_documents)} document(s) from the Markdown file into the vector store.")