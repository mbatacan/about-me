# %%
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec, PodSpec, Pinecone
import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from src.fields import (
    PINECONE_ENVIRONMENT,
    PINECONE_API_KEY,
    HF_API_KEY,
    OPENAI_API_KEY,
    USE_SERVERLESS,
    EMBEDDING_SIZE,
)

if USE_SERVERLESS:
    spec = ServerlessSpec(cloud='aws', region='us-west-2')
else:
    spec = PodSpec(environment=PINECONE_ENVIRONMENT)  # type: ignore

# %%
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "about-me"
# Assuming `pc` is your initialized Pinecone client
index = pc.Index(index_name)
# %%
loader = TextLoader("../data/resume.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
# %%

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_API_KEY, model_name="sentence-transformers/all-MiniLM-l6-v2"  # type: ignore
)

docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)


# %% Example search
query = "What type of Data Science Background does Matthew Batacan have?"

search = docsearch.similarity_search(
    query, k=5  # our search query  # return 3 most relevant docs
)
# %%
# completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo', temperature=0.0
)

qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
)
# %%
result = qa.run(query)
result
# %%
