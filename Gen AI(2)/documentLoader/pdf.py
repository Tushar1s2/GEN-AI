from langchain_community.document_loaders import PyPDFLoader

data=PyPDFLoader("documentLoader/notes.pdf")
docs=data.load();

print(docs)
print(len(docs));