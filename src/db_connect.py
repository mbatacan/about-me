from pinecone import Pinecone, ServerlessSpec, PodSpec
import time
import src.fields as f


class DBConnect:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.pc = Pinecone(api_key=self.api_key)
        self.create_pc_db()

    def create_pc_db(self):
        # configure client
        if f.USE_SERVERLESS:
            spec = ServerlessSpec(cloud='aws', region='us-west-2')
        else:
            # if not using a starter index, you should specify a pod_type too
            spec = PodSpec(environment=f.PINECONE_ENVIRONMENT)  # type: ignore

        # check for and delete index if already exists
        index_name = 'about-me'
        if index_name in self.pc.list_indexes().names():
            self.pc.delete_index(index_name)

        # create a new index
        self.pc.create_index(
            index_name,
            dimension=f.EMBEDDING_SIZE,  # dimensionality of text-embedding-ada-002
            metric='dotproduct',
            spec=spec,
        )

        # wait for index to be initialized
        while not self.pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        return self.pc.describe_index(index_name)
