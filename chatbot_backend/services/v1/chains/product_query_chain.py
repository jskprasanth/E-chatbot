import os
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.runnables import  RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_ollama import OllamaEmbeddings
import os
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from neo4j import  Driver



product_details_chat_template_str = """
Your job is to use the provided employee data to answer 
questions about their roles, performance, and experiences 
within the company. Use the following context to answer questions. 
Be as detailed as possible, but don't make up any information that's 
not from the context. If you don't know an answer, say you don't know.
{context}
"""
product_details_chat_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=product_details_chat_template_str
    )
)
human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="Can you provide details on: {question}?"
    )
)
messages = [product_details_chat_system_prompt, human_prompt]

qa_prompt = ChatPromptTemplate(
    messages=messages,
    input_variables=["context", "question"]
)
llm = OllamaFunctions(model="llama3.1")

# We'll need to have the index created in the ETL process
existing_index = Neo4jVector.from_existing_index(
    OllamaEmbeddings(),
    url=os.environ.get("NEO4J_URL"),
    username=os.environ.get("NEO4J_USER"),
    password=os.environ.get("NEO4J_PASSWORD"),
    index_name="product",
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=existing_index.as_retriever(),
    # ['stuff', 'map_reduce', 'refine', 'map_rerank']
    chain_type="stuff",
)
qa_chain.combine_documents_chain.llm_chain.prompt = qa_prompt
