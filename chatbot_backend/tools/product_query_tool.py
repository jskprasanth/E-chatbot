from langchain.tools import tool

from services.v1.chains.product_query_chain import (
    qa_chain as product_query_chain
)


@tool("product-qa-tool", return_direct=True)
def product_qa_tool(query: str) -> str:
    """Useful for answering questions about Product made by the branded company"""
    response = product_query_chain.invoke(query)

    return response.get("result")