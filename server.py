import os

from fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_community.document_loaders import GithubFileLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

_STR_SEPARATION = "\n" * 8 + "-" * 180 + "\n" * 8


def print_docs(documents):
    for doc in documents:
        print(doc.metadata["path"])
        print(doc.page_content)
        print(_STR_SEPARATION)


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

    @mcp.tool
    def add_repo_to_vector_store(repo_name: str) -> str:
        """Add a repository to the vector store"""

        # delete everything to avoid duplicates, this enables pulling the latest version of the repo
        _delete_repo_from_vector_store(repo_name)

        print(f"Adding {repo_name} to vector store")

        python_code_loader = GithubFileLoader(
            repo=repo_name,  # the repo name
            branch="main",  # the branch name
            access_token=os.environ["GITHUB_ACCESS_TOKEN"],
            github_api_url="https://api.github.com",
            file_filter=lambda file_path: file_path.endswith(".py"),  # load all markdowns files.
        )

        print("Loading docs...")

        docs = python_code_loader.load()

        print(f"Finished loading {len(docs)} docs")

        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=4000, chunk_overlap=1000
        )

        print("Splitting docs...")

        chunks = python_splitter.split_documents(docs)

        print(f"Finished splitting {len(chunks)} chunks")

        for chunk in chunks:
            chunk.metadata["repo_name"] = repo_name

        _ = vector_store.add_documents(documents=chunks)

        message = f"✅ Successfully stored {len(chunks)} chunks in vector store"
        print(message)
        return message

    @mcp.tool
    def delete_repo_from_vector_store(repo_name: str) -> None:
        """Delete a repository from the vector store"""
        _delete_repo_from_vector_store(repo_name)

    def _delete_repo_from_vector_store(repo_name: str) -> None:
        """Delete a repository from the vector store"""
        print(f"Deleting {repo_name} from vector store")
        _ = vector_store.delete(where={"repo_name": repo_name})
        print(f"✅ Successfully deleted {repo_name} from vector store")

    mcp.run(transport="http", host="0.0.0.0", port=9000)


if __name__ == "__main__":
    main()
