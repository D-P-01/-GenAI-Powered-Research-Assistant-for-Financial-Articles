### Demo Video

<video width="600" controls>
  <source src="AI_RESEARCH_ASSISTANT_DEMO_VIDEO.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>




# 🧠 GenAI-Powered Research Assistant for Financial Articles

A cutting-edge **Generative AI** tool that leverages **LLMs** and **Retrieval-Augmented Generation (RAG)** to help **analysts, researchers, and investors** extract meaningful insights from online financial articles in real time.

The system scrapes article content from given URLs, vectorizes it using **FAISS** and **HuggingFace Embeddings**, and allows users to ask questions — with accurate responses retrieved via **LangChain's RetrievalQA pipeline** and an **Ollama-hosted LLaMA3 model**.

---

## 🚀 Features

### 🔗 URL-Based Article Loading
- Users can input URLs of research papers, financial blogs, news, or articles via a Streamlit sidebar.

### 📄 Unstructured Data Handling
- Uses `UnstructuredURLLoader` to scrape and clean content from web pages.

### 🧩 Intelligent Text Chunking
- Splits long articles into manageable chunks using `RecursiveCharacterTextSplitter` for optimized embedding.

### 🔍 Semantic Search using FAISS
- Converts article chunks into embeddings using `BAAI/bge-large-en` and indexes them in a **FAISS** vector store.

### 📦 Persistent Vector Indexing
- Saves the embedding index as a `.pkl` file to avoid redundant processing.

### 🧠 Question-Answering with Sources
- Users can input any query, and the tool will:
  - Search the most relevant article chunks
  - Generate an accurate response using **Ollama + LLaMA3**
  - Display **sources** for traceability

### ✅ Streamlit UI
- A simple, interactive interface that supports real-time feedback and debugging.

### 🛠️ Modular & Extensible Codebase
- Ready to scale with:
  - Multiple articles
  - Additional metadata
  - Improved chunking strategies
  - Different embedding or LLM models

---

## 💡 Use Case: Equity Research Automation

### 🧩 Problem
Equity analysts and retail investors spend countless hours reading through earnings reports, market news, and financial blogs. Extracting meaningful insights from multiple sources is **time-consuming and error-prone**.

### ✅ Solution with This Tool
This assistant can:
- 🔗 Ingest article links per stock (e.g., earnings previews, broker reports)
- 🧠 Summarize key insights
- ❓ Answer targeted questions like:
  - “What are the company’s revenue projections?”
  - “What risks are mentioned in the article?”
  - “Which sectors does the report highlight?”

### 🎯 Benefits
- ⏱️ Saves hours of manual reading
- 📌 Provides accurate, **source-traceable** insights
- 💡 Empowers **faster, better decision-making**

---

## ⚙️ Tech Stack

| Component      | Tool Used                        |
|----------------|----------------------------------|
| **LLM Backend**  | `Ollama` (Local, LLaMA3 70B)     |
| **Framework**    | `LangChain`                      |
| **UI**           | `Streamlit`                      |
| **Embeddings**   | `HuggingFace: BAAI/bge-large-en` |
| **Vector Store** | `FAISS`                          |
| **Scraping**     | `UnstructuredURLLoader`          |
