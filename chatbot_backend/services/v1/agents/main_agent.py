from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent
)
from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_openai import ChatOpenAI
from tools import product_qa_tool, general_qa_tool

agent_prompt = hub.pull("hwchase17/openai-functions-agent")

tools = [product_qa_tool, general_qa_tool]

# Create LLM
llm = OllamaFunctions(model="llama3.1")

# Construct the tool calling agent
agent = create_tool_calling_agent(llm, tools, agent_prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    return_intermediate_steps=True
)