from typing import Any
from src.query import AboutMeBot
from src.etl import ETL
from src.db_connect import DBConnect
import src.fields as f
import os


class ChatClient:
    def __init__(self) -> None:
        client = DBConnect(f.PINECONE_API_KEY)  # type: ignore
        index_name = "about-me"
        index = client.pc.Index(index_name)
        # print working directory
        file_path = os.path.join(os.path.dirname(__file__), '../data/about_me.txt')
        vdb = ETL(file_path=file_path)
        # Initialize chatbot
        qa = AboutMeBot(vdb.vec_store)

        self.qa = qa
        self.index = index
        self.vdb = vdb
        return

    def respond(self, question: str) -> Any:
        response = self.qa.query(question)
        return response
