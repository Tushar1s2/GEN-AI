from langchain_community.retrievers import ArxivRetriever

retriever=ArxivRetriever(
    load_max_docs=3,
    load_all_available_meta=True
)
docs=retriever.invoke("Large language model");

for i, doc in enumerate(docs):
    print("Title:\n",doc.metadata.get("Title"));
    print("Summary\n",doc.page_content);