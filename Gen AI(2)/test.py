from langchain_ollama import OllamaEmbeddings

embeddingModel = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

result = embeddingModel.embed_query("This is a test sentence.")

print("Embedding dimensions:", len(result))
print("First 5 values:", result[:5])