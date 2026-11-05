from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader

data=PyPDFLoader("TextSplitter/notes.pdf");
docs=data.load();

splitter=TokenTextSplitter(
   chunk_size= 1000,
   chunk_overlap=1
)

chunk=splitter.split_documents(docs);
print(chunk[0].page_content)
print(len(chunk));
