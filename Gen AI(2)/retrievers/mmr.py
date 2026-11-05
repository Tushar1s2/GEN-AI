from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

 
docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

embeddingModel = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

vectorStore=Chroma.from_documents(
    documents=docs,
    embedding=embeddingModel
);

#Similarity
similarity_retreiver=vectorStore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

#Mmr
similarity_retreiver2=vectorStore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)



