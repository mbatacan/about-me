from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

import src.fields as f

STUFF_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Use the following pieces of context to answer the question at the "
                "end. If you don't know the answer, just say that you don't know, "
                "don't try to make up an answer.\n\n{context}"
            ),
        ),
        ("human", "{question}"),
    ]
)


def _format_docs(docs) -> str:
    """Join retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


class AboutMeBot:
    def __init__(self, vdb) -> None:
        self.vdb = vdb
        self.initial_prompt = f"You are acting as {f.NAME}.\
            You have a professional at answering questions to tailor to a specific company.\
            You are writing engaging answers to questions and comments given to you. \
            With the information given to you, you will respond relating those previous experiences to the question given.\
            Please answer all questions in first person as if you were {f.NAME}. \
            When you answer, be sure to provide a detailed response including examples from your past experiences.\
            Below I will outline sample questions you might get:\
            you will be asked questions like 'Why would you like to work at company X?' and you will respond with 'I would like to work at company X because...'\
            \
            'What are your strengths?' and you will respond with 'My strengths are...'\
            'What are your weaknesses?' and you will respond with 'My weaknesses are...'\
            'What about this role excites you?' and you will respond with 'This role excites me because...'\
            'What about company X excites you?' and you will respond with 'Company X excites me because...'"
        self.qa = self._chat_bot_init()

    def _chat_bot_init(self):
        llm = ChatOpenAI(
            openai_api_key=f.OPENAI_API_KEY, model_name='gpt-3.5-turbo', temperature=0.0
        )

        retriever = self.vdb.as_retriever()
        chain = (
            {
                "context": retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | STUFF_PROMPT
            | llm
            | StrOutputParser()
        )
        return chain

    def _add_new_docs(self, text: str) -> PineconeVectorStore:
        """
        Takes in raw text to be added to the current vec_store
        """
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.create_documents([text])
        self.vdb.add_documents(docs)
        return self.vdb

    def query(self, question: str) -> dict:
        # Prepend the initial prompt to the user's question
        full_query = self.initial_prompt + " " + question

        # Invoke the QA chain with the full query
        answer = self.qa.invoke(full_query)

        return {"query": full_query, "result": answer}
