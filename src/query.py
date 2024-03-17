from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import src.fields as f


class AboutMeBot:
    def __init__(self, vdb) -> None:
        self.vdb = vdb
        self.qa = self._chat_bot_init()

    def _chat_bot_init(self):
        llm = ChatOpenAI(
            openai_api_key=f.OPENAI_API_KEY, model_name='gpt-3.5-turbo', temperature=0.0
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=self.vdb.as_retriever()
        )
        return qa

    def query(self, question: str) -> str:
        result = self.qa.invoke(question)
        return result
