
import streamlit as st

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


# ---------------- RAG COMPONENTS ----------------

@st.cache_resource
def create_rag_system():

    embeddingsModel = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    vectorStore = Chroma(
        persist_directory="Chroma-db",
        embedding_function=embeddingsModel
    )

    retriever = vectorStore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatOllama(
        model="gemma3:1b"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful, accurate, and context-aware AI assistant.

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

                Your goal: Provide reliable, context-grounded answers without hallucinating or introducing unsupported information."""
            ),
            (
                "human",
                """Context: {context}

                Question: {question}"""
            )
        ]
    )

    return retriever, llm, prompt


# ---------------- INITIALIZE SESSION ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("⚙️ Settings")

    st.markdown("---")

    st.markdown("### Model Information")

    st.write("**LLM:** `gemma3:1b`")
    st.write("**Embedding:** `nomic-embed-text`")
    st.write("**Vector DB:** `Chroma`")
    st.write("**Retrieval:** `MMR`")

    st.markdown("---")

    st.markdown("### Retrieval Configuration")

    st.write("Retrieved chunks: 4")
    st.write("Candidate chunks: 10")
    st.write("Lambda: 0.5")

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Runs locally using Ollama.")


# ---------------- MAIN UI ----------------

st.title("🤖 Local RAG Assistant")

st.markdown(
    "Ask questions about your indexed documents. "
    "Answers are generated using the retrieved context."
)

st.markdown("---")


# ---------------- CHAT HISTORY ----------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------- USER INPUT ----------------

query = st.chat_input("Ask a question about your document...")


if query:

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Searching documents and generating answer..."):

            try:

                retriever, llm, prompt = create_rag_system()

                # Retrieve relevant chunks using MMR
                docs = retriever.invoke(query)

                # Combine retrieved document chunks
                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

                # Create final prompt
                final_prompt = prompt.invoke(
                    {
                        "context": context,
                        "question": query
                    }
                )

                # Generate answer
                response = llm.invoke(final_prompt)

                answer = response.content

                st.markdown(answer)

                # Save assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(
                    f"An error occurred: {e}"
                )
