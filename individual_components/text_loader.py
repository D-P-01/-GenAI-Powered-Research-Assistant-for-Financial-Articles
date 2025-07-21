# from langchain.document_loaders import TextLoader

# loader=TextLoader("nvda.txt")
# data=loader.load()
# print(data[0].page_content)

from langchain_community.document_loaders import UnstructuredURLLoader

loader=UnstructuredURLLoader(
    urls=[
        "https://www.pinecone.io/learn/vector-database/"
        ]
)
data=loader.load()
print(data[0].metadata)