# Import Liberary
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

## Build Knowledge base ----------------------------------------------------------------------------

#1 Load externel knoledege
# Normally in Python, the backslash \ is treated as an escape character (\n = newline)(\t = tab)
# When you add r before the string, Python treats backslashes literally(literal character = takes exactly as you typed)
load_dotenv()
PDF_PATH = os.getenv("PDF_PATH", "./coffee_recipes.pdf")
documents =  PyPDFLoader(PDF_PATH).load()

#2 Split in chunk
# RecursiveCharacterTextSplitter => is the engine/tool that knows how to break text apart
# split_documents()=> is just a method you call on that splitter — it loops through your Documents and tells the splitter to apply its rules to each one
# recursive = something that repeats the same logic step by step, drilling deeper each time if chunk_size=20
# Try to split by paragraphs → still too big.
# Try to split by sentences → still too big.
# Try to split by words → still too big.
# Finally split by characters into chunks of ≤ 20.
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
docs = text_splitter.split_documents(documents)

#3 Generate and store Embedding
# Embeddings → turns text into math (vectors).
# Vectorstore (Chroma) → stores these vectors + metadata for fast similarity search.
# Retriever → queries the vectorstore to fetch the best matching chunks when a user asks something.
embeddings = OllamaEmbeddings(model="llama3.2")
vectorstore = Chroma.from_documents(docs, embedding = embeddings) # pass each docs through embeddings model → get vectors and Store them in a Chroma vector database
retriever = vectorstore.as_retriever(search_kwargs = {"k":5})

## Prompt ----------------------------------------------------------------------------
prompt = PromptTemplate(
    input_variables=["context", "question", "history"],
    template="""
    you are the helpful assistant. Use the conversation history and the context below to answer the question.

    History:
    {history}

    Context:
    {context}

    Question:
    {question}

    Answer clearly:
    """
)

## Define LLM mode ----------------------------------------------------------------------------
LLM = OllamaLLM(model="llama3.2")

## Parser----------------------------------------------------------------------------
Parser = StrOutputParser()

## Chain----------------------------------------------------------------------------
chain = (
    {
        "context": retriever | (lambda docs: "\n".join([d.page_content for d in docs])), # It return concatenated text of the top-K retrieved chunks
        "question": lambda x: x,                                                         # It take x and return x
        "history": lambda _: "\n".join(                                                  # Build conversation history
            [f"User: {msg['user']}\nAssistant: {msg['bot']}" for msg in st.session_state.get("chat_history", [])]
        )
    }
    | prompt
    | LLM
    | Parser
)

## Streamlit UI----------------------------------------------------------------------------
st.title("RAG Chatbot (Static PDF + Llama3.2)")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask something from the knowledge base:")

if query:  # query is the text the user typed in Streamlit (st.text_input)
    result = chain.invoke(query)

    # Save chat history
    st.session_state.chat_history.append({"user": query, "bot": result})

    # Show last answer
    st.subheader("Answer:")
    st.write(result)

# Show full chat history
if st.session_state.chat_history:
    st.subheader("Chat History")
    for msg in st.session_state.chat_history:
        st.markdown(f"**🧑 User:** {msg['user']}")
        st.markdown(f"**🤖 Assistant:** {msg['bot']}")
        st.markdown("---")
