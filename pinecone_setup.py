import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()
pinecone_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX_NAME")


class PineconeSetup:
    def __init__(self):
        self.pinecone = Pinecone(api_key=pinecone_key)
        self.index_name = pinecone_index
        self.vector_store = None
    def create_index(self):
        if self.index_name not in self.pinecone.list_indexes().names():
            self.pinecone.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")  # <-- add cloud and region
            )
        print(f"Index '{self.index_name}' is ready.")
    def get_vector_store(self):
        if not self.vector_store:
            self.vector_store = PineconeVectorStore(index_name=self.index_name)
        return self.vector_store
    def insert_data(self, data):
        vector_store = self.get_vector_store()
        for item in data:
            vector_store.add(item['id'], item['vector'], metadata=item.get('metadata', {}))
        print(f"Inserted {len(data)} items into the vector store.")