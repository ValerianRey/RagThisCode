import argparse
import os

from langchain_community.document_loaders import GithubFileLoader
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from common import get_vector_store


def add_repo_to_vector_store(repo_name: str, vector_store: VectorStore, branch: str = "main"):
    """Add a repository to the vector store

    Args:
        repo_name: The repository name in format 'owner/repo'
        branch: The branch name to load from (default: main)
        vector_store: The vector store to add the repository to
    """

    # delete everything to avoid duplicates, this enables pulling the latest version of the repo
    _delete_repo_from_vector_store(repo_name, vector_store)

    print(f"Adding {repo_name} to vector store")

    python_code_loader = GithubFileLoader(
        repo=repo_name,  # the repo name
        branch=branch,  # the branch name
        access_token=os.environ["GITHUB_ACCESS_TOKEN"],
        github_api_url="https://api.github.com",
        file_filter=lambda file_path: file_path.endswith(".py"),
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


def _delete_repo_from_vector_store(repo_name: str, vector_store: VectorStore) -> None:
    """Delete a repository from the vector store"""
    print(f"Deleting {repo_name} from vector store")
    vector_store.delete(where={"repo_name": repo_name})
    print(f"✅ Successfully deleted {repo_name} from vector store")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Store code from a GitHub repository in a vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            %(prog)s TorchJD/torchjd
            %(prog)s TorchJD/torchjd --branch main
        """,
    )

    parser.add_argument(
        "repo_name", help="Repository name in format 'owner/repo' (e.g., 'TorchJD/torchjd')"
    )

    parser.add_argument("--branch", default="main", help="Branch name to load from (default: main)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    vector_store = get_vector_store()
    add_repo_to_vector_store(args.repo_name, vector_store, args.branch)
