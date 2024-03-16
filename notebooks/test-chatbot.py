# %%
from src.db_connect import DBConnect
from src.etl import ETL
import src.fields as f
from src.query import AboutMeBot

# %% Client Connect
client = DBConnect(f.PINECONE_API_KEY)  # type: ignore
index_name = "about-me"
# Assuming `pc` is your initialized Pinecone client
index = client.pc.Index(index_name)
# %% ETL
vdb = ETL('../data/resume.txt')
qa = AboutMeBot(vdb.vec_store)
# %%
query = "What type of Data Science Background I have?"

# Example search
result = qa.query(query)
print(result)
# %%
