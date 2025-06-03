import os
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter



gpt_key = os.getenv("GPT_API_KEY")
gpt_model = os.getenv("OPENAI_MODEL")
embedding_model = os.getenv("EMBEDDING_MODEL")

# Load documents
loaders = [
    TextLoader("paul_graham_essay.txt", encoding="utf-8"),
    TextLoader("state_of_the_union.txt", encoding="utf-8"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Set up embedding model
embedding = OpenAIEmbeddings(api_key=gpt_key, model=embedding_model)

# Define text splitters
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# Initialize Chroma vector store (persisted locally)
vectorstore = Chroma(
    collection_name="parent_docs",
    embedding_function=embedding,
    persist_directory="./chroma_db"
)

# Initialize in-memory store for parent documents
store = InMemoryStore()

# Initialize ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Add documents to retriever
retriever.add_documents(docs)

# Perform a similarity search
query = "justice breyer"
retrieved_docs = retriever.invoke(query)

# Output the content of the first retrieved document
print(retrieved_docs[0].page_content)
