import os
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone_setup import PineconeSetup


loaders = [
    TextLoader("paul_graham_essay.txt"),
    TextLoader("state_of_the_union.txt"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
    
    
pinecone_key = os.getenv("PINECONE_API_KEY")

gpt_key = os.getenv("GPT_API_KEY")
gpt_model = os.getenv("OPENAI_MODEL")
embedding_model = os.getenv("EMBEDDING_MODEL")

embedding = OpenAIEmbeddings(api_key=gpt_key, model=embedding_model)

child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

db_setup = PineconeSetup().create_index()

vectordb = PineconeVectorStore.from_documents(
    documents=splits,
    embedding=embedding,
    index_name=os.getenv("PINECONE_INDEX_NAME"),
)
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=OpenAIEmbeddings()
)
# The storage layer for the parent documents
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)