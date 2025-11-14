from fastapi import Request
from fastmcp import FastMCP
from langchain_core.vectorstores import VectorStore
from starlette.responses import JSONResponse

from common import get_vector_store
from store_code_in_vector_db import add_repo_to_vector_store

_STR_SEPARATION = "\n" * 8 + "-" * 180 + "\n" * 8


def print_docs(documents):
    for doc in documents:
        print(doc.metadata["path"])
        print(doc.page_content)
        print(_STR_SEPARATION)


def run_mcp_server(repo_name: str, vector_store: VectorStore, server_ip: str, server_port: int):
    """Run the MCP server (blocking call) at the given IP and port"""

    mcp = FastMCP("Demo 🚀")

    @mcp.tool
    def similarity_search(query: str) -> list[str]:
        """Search for similar code snippets in the vector store"""
        retrieved_docs = vector_store.similarity_search(query)
        return [doc.page_content for doc in retrieved_docs]

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request):
        return JSONResponse({"status": "healthy", "service": "mcp-server", "repo_name": repo_name})

    mcp.run(transport="http", host=server_ip, port=server_port)


def start_server(repo_name: str, server_ip: str, server_port: int):
    """Legacy function for backwards compatibility"""
    vector_store = get_vector_store()
    add_repo_to_vector_store(repo_name, vector_store)
    run_mcp_server(repo_name, vector_store, server_ip, server_port)


if __name__ == "__main__":
    repo_name = "TorchJD/torchjd"
    server_ip = "54.36.102.143"  # "0.0.0.0" when exposed to the internet, localhost when not
    server_port = 9000
    start_server(repo_name, server_ip, server_port)
