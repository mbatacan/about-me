from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import src.fields as f


class AboutMeBot:
    def __init__(self, vdb) -> None:
        self.vdb = vdb
        self.initial_prompt = "You are a professional job-hunting counselor. You will be helping to answer questions for your client. Answer with the details about the client you are given in the vector store given."
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
        # Prepend the initial prompt to the user's question
        full_query = self.initial_prompt + " " + question

        # Invoke the QA chain with the full query
        result = self.qa.invoke(full_query)

        return result
