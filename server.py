from fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

str_separation = "\n" * 8 + "-" * 180 + "\n" * 8
def print_docs(documents):
    for doc in documents:
        print(doc.metadata["path"])
        print(doc.page_content)
        print(str_separation)


def main():

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_store = Chroma(
        collection_name="torchjd_code_collection",
        embedding_function=embeddings,
        persist_directory="./data/chroma_langchain_db",
    )
    
    mcp = FastMCP("Demo 🚀")

    @mcp.tool
    def similarity_search(query: str) -> list[str]:
        """Search for similar code snippets in the vector store"""
        retrieved_docs = vector_store.similarity_search(query)
        return [doc.page_content for doc in retrieved_docs]
   
    mcp.run(transport="http", host="127.0.0.1", port=9000)
    


if __name__ == "__main__":
    main()