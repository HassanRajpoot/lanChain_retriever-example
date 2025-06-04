import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import Literal, List
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START
from pprint import pprint



load_dotenv()

# Embeddings
embd = OpenAIEmbeddings(model="text-embedding-3-small")

# Source URLs
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load documents
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Vectorstore
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embd,
)
retriever = vectorstore.as_retriever()

### ROUTER ###

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"] = Field(
        ..., description="Choose web search or vectorstore."
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
Use the vectorstore for questions about agents, prompt engineering, or adversarial attacks. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
question_router = route_prompt | structured_llm_router

### RETRIEVAL GRADER ###

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Relevant to question: 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeDocuments)
retrieval_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing relevance of a document to a user question. Answer with 'yes' or 'no'."),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
])
retrieval_grader = retrieval_grader_prompt | structured_llm_grader

### RAG CHAIN ###

prompt = hub.pull("rlm/rag-prompt")
rag_chain = prompt | llm | StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

### HALLUCINATION GRADER ###

class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Grounded in facts: 'yes' or 'no'")

hallucination_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", "Is the generation grounded in the facts? Respond 'yes' or 'no'."),
    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
])
hallucination_grader = hallucination_grader_prompt | llm.with_structured_output(GradeHallucinations)

### ANSWER GRADER ###

class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Answers the question: 'yes' or 'no'")

answer_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", "Does the generation answer the question? Respond 'yes' or 'no'."),
    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
])
answer_grader = answer_grader_prompt | llm.with_structured_output(GradeAnswer)

### QUESTION REWRITER ###

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the input question to improve vectorstore retrieval."),
    ("human", "Initial question: \n\n {question} \n Reformulate:")
])
question_rewriter = rewrite_prompt | llm | StrOutputParser()

### WEB SEARCH ###

web_search_tool = TavilySearchResults(k=3)

### GRAPH STATE ###

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]

### FUNCTIONS ###

def retrieve(state):
    print("---RETRIEVE---")
    docs = retriever.invoke(state["question"])
    return {"documents": docs, "question": state["question"]}

def generate(state):
    print("---GENERATE---")
    generation = rag_chain.invoke({"context": state["documents"], "question": state["question"]})
    return {"documents": state["documents"], "question": state["question"], "generation": generation}

def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score.lower() == "yes":
            print("---RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---NOT RELEVANT---")
    return {"documents": filtered_docs, "question": question}

def transform_query(state):
    print("---TRANSFORM QUERY---")
    better_question = question_rewriter.invoke({"question": state["question"]})
    return {"documents": state["documents"], "question": better_question}

def web_search(state):
    print("---WEB SEARCH---")
    results = web_search_tool.invoke({"query": state["question"]})
    combined = "\n".join([r["content"] for r in results])
    return {"documents": [Document(page_content=combined)], "question": state["question"]}

def route_question(state):
    print("---ROUTE QUESTION---")
    route = question_router.invoke({"question": state["question"]})
    return "web_search" if route.datasource == "web_search" else "vectorstore"

def decide_to_generate(state):
    print("---ASSESS FILTERED DOCUMENTS---")
    return "generate" if state["documents"] else "transform_query"

def grade_generation_v_documents_and_question(state):
    print("---GRADE GENERATION---")
    h_score = hallucination_grader.invoke({"documents": state["documents"], "generation": state["generation"]})
    if h_score.binary_score.lower() == "yes":
        a_score = answer_grader.invoke({"question": state["question"], "generation": state["generation"]})
        return "useful" if a_score.binary_score.lower() == "yes" else "not useful"
    else:
        return "not supported"

### BUILD GRAPH ###

workflow = StateGraph(GraphState)
workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

workflow.add_conditional_edges(START, route_question, {
    "web_search": "web_search",
    "vectorstore": "retrieve"
})
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate, {
    "transform_query": "transform_query",
    "generate": "generate"
})
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges("generate", grade_generation_v_documents_and_question, {
    "not supported": "generate",
    "useful": END,
    "not useful": "transform_query"
})

app = workflow.compile()

### RUN ###

inputs = {"question": "What player at the Bears expected to draft first in the 2024 NFL draft?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")
    pprint("\n---\n")

pprint("Final answer:")
pprint(value["generation"])
