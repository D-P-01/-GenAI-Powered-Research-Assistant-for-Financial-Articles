# UPDATED CODE WITH DEBUGGING STATEMENTS

import os 
import streamlit as st
import pickle
import time
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

llm = Ollama(model="llama3.3:70b", base_url="http://192.168.10.41:11434")
print("[INFO] Ollama LLM initialized.")

st.title("New Research Tool 📈")
st.sidebar.title("New Article URLs")

urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url.strip())

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "/home/dhanjay/Documents/Personal/LangChain_Learning/equity_research_tool/individual_components/vectorindex2.pkl"

main_placeholder = st.empty()
answer_placeholder = st.empty()  # For showing answers separately

if process_url_clicked:
    if os.path.exists(file_path):
        main_placeholder.success("✅ Embedding already exists. Skipping processing...")
        print("[INFO] Embedding already exists. Skipping processing...")
    else:
        # Load data
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading Started...✅✅✅")
        print("[INFO] Loading documents from URLs...")
        # data = loader.load()
        data = loader.load()
        for doc, url in zip(data, urls):
            doc.metadata["source"] = url


        # Splitting data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=1000,
            chunk_overlap=200
        )
        main_placeholder.text("Text Splitter Started...✅✅✅")
        docs = text_splitter.split_documents(data)
        print(f"[INFO] Text split into {len(docs)} chunks.")

        # Create embeddings and save to FAISS index
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en",
            encode_kwargs={"normalize_embeddings": True}
        )
        vectorstore_huggingface = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding vector started building...✅✅✅")
        print("[INFO] Embeddings created.")
        time.sleep(2)

        # Save FAISS index
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_huggingface, f)
        print(f"[INFO] Embeddings saved to {file_path}")

# Always show question box
query = st.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        print("[INFO] Running query on LLM...")
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k":10},
            verbose=True)
        )
        result = chain({"question": query}, return_only_outputs=True)

        print("[INFO] LLM response received.")
        print(f"[DEBUG] Answer: {result['answer']}")
        print(f"[DEBUG] Sources: {result.get('sources', 'N/A')}")

        st.header("Answer:")
        answer_placeholder.success(result["answer"])
        st.markdown(f"**Sources:** {result.get('sources', 'Not available')}")


# ORIGINAL CODE
# import os 
# import streamlit as st
# import pickle
# import time
# from langchain_community.llms import  Ollama
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS

# llm=Ollama(model="llama3.3:70b",
#            base_url="http://192.168.10.41:11434")

# st.title("New Research Tool 📈")
# st.sidebar.title("New Article URLs")


# urls=[]

# for i in range(2):
#     url=st.sidebar.text_input(f"URL {i+1}")
#     urls.append(url)

# process_url_clicked=st.sidebar.button("Process URLs")
# file_path="/home/dhanjay/Documents/Personal/LangChain_Learning/equity_research_tool/individual_components/vectorindex.pkl"

# main_placeholder=st.empty()

# if process_url_clicked:
#     if os.path.exists(file_path):
#         main_placeholder.text("Embedding already exists. Skipping processing...✅✅✅")
#     else:
#         # Load data
#         loader=UnstructuredURLLoader(urls=urls)
#         main_placeholder.text("Data Loading Started...✅✅✅")
#         data=loader.load()

#         # Splitting data
#         text_splitter=RecursiveCharacterTextSplitter(
#             separators=["\n\n", "\n", ".", ","],
#             chunk_size=500
#         )
#         main_placeholder.text("Text Splitter Started...✅✅✅")
#         docs=text_splitter.split_documents(data)

#         # Create embeddings and save it to FAISS index
#         embeddings = HuggingFaceEmbeddings(
#             model_name="BAAI/bge-large-en",
#             encode_kwargs={"normalize_embeddings": True}
#         )
#         vectorstore_huggingface=FAISS.from_documents(docs,embeddings)
#         main_placeholder.text("Embedding vector started building...✅✅✅")
#         time.sleep(2)

#         # Save FAISS index to a pickle file
#         with open(file_path,"wb") as f:
#             pickle.dump(vectorstore_huggingface,f)
    
#     query=main_placeholder.text_input("Question: ")
#     if query:
#         if os.path.exists(file_path):
#             with open(file_path,"rb") as f:
#                 vectorstore=pickle.load(f)
#                 chain=RetrievalQAWithSourcesChain.from_llm(
#                     llm=llm,
#                     retriever=vectorstore.as_retriever(search_kwargs={"k": 10})
#                 )
#                 result=chain({"question":query},return_only_outputs=True) #Result format {"answer":"whatever is the ans", "sources":""}
#                 st.header("Answer:")
#                 st.subheader(result["answer"])




