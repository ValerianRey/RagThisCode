import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import GithubFileLoader
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


def main():
    """Add a repository to the vector store"""

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_store = Chroma(
        collection_name="torchjd_code_collection",
        embedding_function=embeddings,
        persist_directory="./data/chroma_langchain_db",
    )

    repo_name = "TorchJD/torchjd"

    # delete everything to avoid duplicates, this enables pulling the latest version of the repo
    _delete_repo_from_vector_store(repo_name, vector_store)

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

    vector_store.add_documents(documents=chunks)

    message = f"✅ Successfully stored {len(chunks)} chunks in vector store"
    print(message)

    repo_name = "TorchJD/torchjd"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

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

    vector_store = Chroma(
        collection_name="code_collection",
        embedding_function=embeddings,
        persist_directory="./data/chroma_langchain_db",
    )
    _ = vector_store.add_documents(documents=chunks)

    print(f"✅ Successfully stored {len(chunks)} chunks in vector store")


def _delete_repo_from_vector_store(repo_name: str, vector_store: VectorStore) -> None:
    """Delete a repository from the vector store"""
    print(f"Deleting {repo_name} from vector store")
    vector_store.delete(where={"repo_name": repo_name})
    print(f"✅ Successfully deleted {repo_name} from vector store")


if __name__ == "__main__":
    main()
