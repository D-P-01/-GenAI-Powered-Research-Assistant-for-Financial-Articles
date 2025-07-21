# import os
# import pickle
# import time
# from langchain_community.llms import Ollama
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.docstore.document import Document

# # Load LLM
# llm = Ollama(
#     model='llama3.3:70b',
#     base_url='http://192.168.10.41:11434',
#     temperature=0.7
# )
# print("✅ LLM Loaded")

# # File paths
# index_path = "vectorindex2.pkl"
# docs_path = "docs.pkl"

# # Check if FAISS index exists
# if os.path.exists(index_path):
#     print("✅ Found saved FAISS index. Loading...")
#     with open(index_path, "rb") as f:
#         vectorindex2 = pickle.load(f)
# else:
#     print("🆕 FAISS index not found. Creating new one...")

#     # Load content from URL(s)
#     urls = [
#         # "https://www.carwale.com/mercedes-benz-cars/",
#         "https://en.wikipedia.org/wiki/Elon_Musk"
#     ]
#     loader = UnstructuredURLLoader(urls=urls)
#     data = loader.load()
#     print(f"📄 Loaded {len(data)} documents")

#     # Clean and filter documents
#     data = [doc for doc in data if doc.page_content and len(doc.page_content.strip()) > 50]
#     for doc in data:
#         doc.page_content = doc.page_content.replace("\n", " ").strip()

#     # Split into chunks
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     docs = splitter.split_documents(data)
#     print(f"🔗 Split into {len(docs)} chunks")

#     # Save docs for reuse (optional)
#     with open(docs_path, "wb") as f:
#         pickle.dump(docs, f)

#     # Load embeddings model
#     embeddings = HuggingFaceEmbeddings(
#         model_name="BAAI/bge-large-en",
#         encode_kwargs={"normalize_embeddings": True}
#     )

#     # Extract plain text from docs
#     texts = [doc.page_content for doc in docs]

#     # Batch embed documents
#     print("📡 Generating embeddings (this may take a while)...")
#     start_time = time.time()
#     text_embeddings = embeddings.embed_documents(texts)
#     print(f"✅ Embeddings generated in {time.time() - start_time:.2f} seconds")

#     # Create FAISS index manually
#     vectorindex2 = FAISS.from_embeddings(
#         embeddings=text_embeddings,
#         documents=docs,
#         embedding=embeddings
#     )
#     print("📦 FAISS index created")

#     # Save FAISS index
#     with open(index_path, "wb") as f:
#         pickle.dump(vectorindex2, f)
#     print("💾 FAISS index saved to disk")

# # Create retrieval-based QA chain
# chain = RetrievalQAWithSourcesChain.from_llm(
#     llm=llm,
#     retriever=vectorindex2.as_retriever(search_kwargs={"k": 10})
# )
# print("🔗 Retrieval QA Chain Ready")

# # Example query
# query = "What is the name of Elon Musk's company? What is his age? What is his yearly income?"
# print(f"\n🤖 Asking: {query}\n")





# ORIGINAL CODE
import os
import pickle
import langchain
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQAWithSourcesChain 
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load LLM
llm = Ollama(
    model='llama3.3:70b',
    base_url='http://192.168.10.41:11434',
    temperature=0.7
)
print("LLM LOADED...")

# Set vector index file path
file_path = "vectorindex2.pkl"

# Check if index file exists
if os.path.exists(file_path):
    print("vectorindex2.pkl already exists. Loading without reprocessing...")
    with open(file_path, "rb") as f:
        vectorindex2 = pickle.load(f)
else:
    print("vectorindex2.pkl not found. Creating new one...")

    # Load content from URL
    loaders = UnstructuredURLLoader(
        urls=["https://www.carwale.com/bmw-cars/",
              "https://www.carwale.com/mercedes-benz-cars/"]
        # urls=["https://en.wikipedia.org/wiki/Elon_Musk"]
    )
    data = loaders.load()
    print(f"UNstructured url o/p: {len(data)}")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(data)
    print(f"Number of chunks: {len(docs)}")

    # create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en",
        encode_kwargs={"normalize_embeddings": True}
    )

    # Create FAISS index
    vectorindex2 = FAISS.from_documents(docs, embeddings)
    print("VECTORINDEX CREATED...")

    # Save index
    with open(file_path, "wb") as f:
        pickle.dump(vectorindex2, f)
    print("VECTORINDEX.pkl SAVED...")

# Create retrieval chain
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorindex2.as_retriever(search_kwargs={"k": 10}))
print("CHAIN CREATED...")

# Run a test query
query = "i want to buy a car in a budget of 50lakhs,suggest me all avaliable options for bmw and mercedes benz?"
langchain.debug = True
response = chain({"question": query}, return_only_outputs=True)
print(response)

























