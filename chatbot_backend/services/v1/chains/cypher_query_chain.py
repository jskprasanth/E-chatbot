import os
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
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

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

graph.refresh_schema()

cypher_generation_template_str = """
Task:
Generate Cypher query for a Neo4j graph database.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}

Note:
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything other than
for you to construct a Cypher statement. Do not include any text except
the generated Cypher statement. Make sure the direction of the relationship is
correct in your queries. Make sure you alias both entities and relationships
properly. Do not run any queries that would add to or delete from
the database. Make sure to alias all statements that follow as with
statement (e.g. WITH c as customer, o.orderID as order_id).
If you need to divide numbers, make sure to
filter the denominator to be non-zero.

Examples:
# Retrieve the total number of orders placed by each customer.
MATCH (c:Customer)-[o:ORDERED_BY]->(order:Order)
RETURN c.customerID AS customer_id, COUNT(o) AS total_orders
# List the top 5 products with the highest unit price.
MATCH (p:Product)
RETURN p.productName AS product_name, p.price AS product_price
ORDER BY product_price DESC
LIMIT 5
# Find all products which is sold most.
MATCH (b:Brand)-[r:Maker]->(p:Product)
RETURN p.name AS s_no, COUNT(o) AS sold_most
String category values:
Use existing strings and values from the schema provided. 
# Get all information about a given order, all its relationships and nodes connected to it.
MATCH (o: Order)-[r]-(n)
WHERE o.orderID = 10249
RETURN o, r, n;

# Get freight cost for each shippment to Germany by Speedy Express.
MATCH (s: Shipper)<-[sr: SHIPPED_BY]-(o: Order)-[i:INCLUDES]->(p: Product)
WHERE s.companyName = "Speedy Express" 
AND o.shipCountry = "Germany"
RETURN  COUNT(*) AS shpments_to_germany,
o.freight AS freight, p.productName AS product_name;

# Get sum of total freight cost for each order by Speedy Express in Germany.
MATCH (s: Shipper)<-[sr: SHIPPED_BY]-(o: Order)-[i:INCLUDES]->(p: Product)
WHERE s.companyName = "Speedy Express" 
AND o.shipCountry = "Germany"
RETURN  COUNT(*) AS shpments_to_germany,
SUM(o.freight) AS freight;



String category values:
Use existing strings and values from the schema provided. 

The question is:
{question}
"""


cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=cypher_generation_template_str
)

qa_generation_template_str = """
You are an assistant that takes the results from a Neo4j Cypher query and forms a human-readable response. The query results section contains the results of a Cypher query that was generated based on a user's natural language question. The provided information is authoritative; you must never question it or use your internal knowledge to alter it. Make the answer sound like a response to the question.

Query Results:
{context}

Question:
{question}

If the provided information is empty, respond by stating that you don't know the answer. Empty information is indicated by: []

If the information is not empty, you must provide an answer using the results. If the question involves a time duration, assume the query results are in units of days unless specified otherwise.

When names are provided in the queary results, such as hospital names, be cautious of any names containing commas or other punctuation. For example, 'Jones, Brown and Murray' is a single hospital name, not multiple hospitals. Ensure that any list of names is presented clearly to avoid ambiguity and make the full names easily identifiable.

Never state that you lack sufficient information if data is present in the query results. Always utilize the data provided.

Your answer should be in a JSON format with the following keys:

    "Answer": "The answer to the question.",
    "Context": Query Results

Helpful Answer:
"""


qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template_str
)

cypher_chain = GraphCypherQAChain.from_llm(
    allow_dangerous_requests=True,
    top_k=10,
    graph=graph,
    verbose=True,
    validate_cypher=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    qa_llm=OllamaFunctions(model="llama3.1", temperature=0),
    cypher_llm=OllamaFunctions(model="llama3.1", temperature=0),
)
