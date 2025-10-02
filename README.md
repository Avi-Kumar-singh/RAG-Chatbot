# RAG-Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with **Llama 3.2**, **Streamlit**, and **ChromaDB**.  
It allows users to ask questions about a PDF document (coffee recipes in this case) and get AI-generated answers with citations from the document.

---

## 🚀 Features
- Upload and chat with PDF documents  
- Uses **LangChain** for document retrieval  
- Embeddings generated via **Ollama**  
- Vector search with **ChromaDB**  
- Interactive UI powered by **Streamlit**  
- Environment variables managed with `.env`  

---

## ⚡ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Avi-Kumar-singh/RAG-Chatbot.git
cd RAG-Chatbot

### 2. Install dependencies
```bash
pip install -r requirements.txt

### 3. Set up environment variables
Create a `.env` file in the root directory and add:
```bash
PDF_PATH=./coffee_recipes.pdf
