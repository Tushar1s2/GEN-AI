#load pdf
#split into chunk
#create the embeddings
#store the chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma



data=PyPDFLoader("documentLoader/book.pdf");
docs=data.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
);

chunks=splitter.split_documents(docs);

embeddingModel = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

vectorStore=Chroma.from_documents(
    documents=chunks,
    embedding=embeddingModel,
    persist_directory="Chroma-db"
);

