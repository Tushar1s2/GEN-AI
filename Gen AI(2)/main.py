from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

embeddingsModel=OllamaEmbeddings(
     model="nomic-embed-text",
     base_url="http://localhost:11434"
);

vectorStore=Chroma(
    persist_directory="Chroma-db",
    embedding_function=embeddingsModel
);

retriever=vectorStore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
);

llm=ChatOllama(
    model="gemma3:1b"
);
  
#prompt
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","""You are a helpful, accurate, and context-aware AI assistant.

            Instructions:

            Answer the user's questions using only the information explicitly available in the provided context.

            Do not use your own knowledge, external sources, assumptions, or guesses to answer questions.

            If the answer is not present in the provided context, respond exactly with: "I could not find the answer in the provided context."

            Do not make up facts or generate information that is not supported by the context.

            If the context contains only partial information, answer only the part that is supported by the context. Do not fill in missing details.

            Keep answers clear, concise, and directly relevant to the user's question.

            If the context is ambiguous or insufficient to determine the correct answer, respond with: "I could not find the answer in the provided context."

            Do not mention information from outside the context, even if you know the answer.

            Treat the provided context as the sole source of truth.

            Your goal: Provide reliable, context-grounded answers without hallucinating or introducing unsupported information. """
        ),
        ("human",""" Context:{context}  Question:{question} """)
    ]
);

print("Rag System created");
print("Press 0 to exit");

while True:
    query=input("You: ");
    if query=="0":
        break
    
    docs=retriever.invoke(query);
    
    context="\n\n".join(
        [doc.page_content for doc in docs]
    );
    
    final_prompt=prompt.invoke({
        "context":context,
        "question":query
    });
    
    response=llm.invoke(final_prompt);
    print(f"\n AI: {response.content}");


