import os
from typing import List
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from pinecone_setup import PineconeSetup

pinecone_key = os.getenv("PINECONE_API_KEY")

gpt_key = os.getenv("GPT_API_KEY")
gpt_model = os.getenv("OPENAI_MODEL")
embedding_model = os.getenv("EMBEDDING_MODEL")

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()
embedding = OpenAIEmbeddings(api_key=gpt_key, model=embedding_model)
llm = ChatOpenAI(api_key=gpt_key, model=gpt_model,temperature=0)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)
db_setup = PineconeSetup().create_index()

vectordb = PineconeVectorStore.from_documents(
    documents=splits,
    embedding=embedding,
    index_name=os.getenv("PINECONE_INDEX_NAME"),
)

# Output parser will split the LLM result into a list of queries
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines


output_parser = LineListOutputParser()

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)


# Chain
llm_chain = QUERY_PROMPT | llm | output_parser

# Other inputs
question = "What are the approaches to Task Decomposition?"

# Run
retriever = MultiQueryRetriever(
    retriever=vectordb.as_retriever(), llm_chain=llm_chain, parser_key="lines"
)  # "lines" is the key (attribute name) of the parsed output

# Results
unique_docs = retriever.invoke("What does the course say about regression?")
print(len(unique_docs))