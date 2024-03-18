from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import src.fields as f
from typing import Union


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
        loader = TextLoader(filepath)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        return docs

    def _embed_docs(self):
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=f.HF_API_KEY, model_name="sentence-transformers/all-MiniLM-l6-v2"  # type: ignore
        )
        vec_store = PineconeVectorStore.from_documents(
            self.docs, embeddings, index_name='about-me'
        )
        return vec_store

    def _add_new_docs(self, text: str) -> PineconeVectorStore:
        """
        Takes in text to be added to current vdb
        """

        return self.vec_store
