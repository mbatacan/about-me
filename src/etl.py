from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

import src.fields as f


class ETL:
    def __init__(self, file_path: str) -> None:
        """
        ETL Object - Takes in file_path: str to text to reference
        """
        self.docs = self._create_docs(file_path)
        self.vec_store = self._embed_docs()

    def _create_docs(self, filepath: str):
        """
        create document objects to be uploaded to vectorstore

        Input - filepath:str
        Output - docs: list[Documents]
        """
        with open(filepath) as fh:
            text = fh.read()
        documents = [Document(page_content=text, metadata={"source": filepath})]
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        return docs

    def _embed_docs(self):
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-l6-v2",
            huggingfacehub_api_token=f.HF_API_KEY,  # type: ignore
        )
        vec_store = PineconeVectorStore.from_documents(
            self.docs, embeddings, index_name='about-me'
        )
        return vec_store
