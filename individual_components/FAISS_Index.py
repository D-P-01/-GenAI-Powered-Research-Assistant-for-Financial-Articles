import pandas as pd 

pd.set_option('display.max_colwidth', 100)
df=pd.read_csv("sample_text.csv")
print(df)

from sentence_transformers import SentenceTransformer
encoder=SentenceTransformer("all-mpnet-base-v2")
vectors=encoder.encode(df.text)
print(vectors.shape)

dim=vectors.shape[1]

import faiss 
index=faiss.IndexFlatL2(dim)
index.add(vectors)

search_query="I want to go for vacation?"
vec=encoder.encode(search_query)
print(vec.shape)

import numpy as np
svec=np.array(vec).reshape(1,-1)
print(svec.shape)

distances,I=index.search(svec,k=2)
print(I)
print(df.loc[I[0]])



