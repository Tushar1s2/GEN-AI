from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


docs = [
    Document(page_content="Pandas is used for data analysis in python.", metadata={"source": "ml.txt"}),
    Document(page_content="Neural network is used in Deep Learning.", metadata={"source": "dl.txt"}),
    Document(page_content="Python Programming is widely used for AI.", metadata={"source": "python.txt"})
]

embedding_model=OllamaEmbeddings(model="nomic-embed-text")

vectorStore=Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
);

result=vectorStore.similarity_search("What is used for data analysis?",k=2);

for r in result:
    print(r)
    
retriever=vectorStore.as_retriever();

docs=retriever.invoke("Explain deep learning");

for d in docs:
    print(d.page_content);  


