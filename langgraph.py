from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

# 1. Define your state
from typing import TypedDict

class GraphState(TypedDict):
    question: str
    answer: str

# 2. Define a function to run the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

def ask_llm(state: GraphState) -> GraphState:
    question = state["question"]
    response = llm.invoke(question)
    return {"question": question, "answer": response.content}

# 3. Create a graph
graph_builder = StateGraph(GraphState)

# Add node
graph_builder.add_node("llm_node", RunnableLambda(ask_llm))

# Define edges
graph_builder.set_entry_point("llm_node")
graph_builder.add_edge("llm_node", END)

# Compile the graph
graph = graph_builder.compile()

# 4. Run the graph
inputs = {"question": "What is the capital of Japan?"}
result = graph.invoke(inputs)

print("Answer:", result["answer"])
