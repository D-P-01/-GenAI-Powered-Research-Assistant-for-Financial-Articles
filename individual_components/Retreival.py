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
    model='---',
    base_url='---',
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

























