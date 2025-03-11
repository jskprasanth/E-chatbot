from langchain.tools import tool

from services.v1.chains.cypher_query_chain import (
    cypher_chain
)


@tool("general-qa-tool", return_direct=True)
def general_qa_tool(query: str) -> str:
    """Useful for answering general questions about product, price, in_stock, gender, and currency."""
    response = cypher_chain.invoke(query)

    return response.get("result")