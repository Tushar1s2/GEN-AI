from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

splitter=CharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=1
)

data=TextLoader("TextSplitter/notes2.txt");
docs=data.load()

chunks=splitter.split_documents(docs);
print(chunks);
print(len(chunks))