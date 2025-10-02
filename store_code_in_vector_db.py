import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import GithubFileLoader, GitHubIssuesLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)


def main():

    print("🔍 Starting to store code in vector store")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    python_code_loader = GithubFileLoader(
        repo="TorchJD/torchjd",  # the repo name
        branch="main",  # the branch name
        access_token=os.environ["GITHUB_ACCESS_TOKEN"],
        github_api_url="https://api.github.com",
        file_filter=lambda file_path: file_path.endswith(".py"),  # load all markdowns files.
    )

    docs = python_code_loader.load()

    print(f"🔍 Loaded {len(docs)} docs")

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size = 4000, chunk_overlap = 1000
    )

    chunks = python_splitter.split_documents(docs)

    print(f"🔍 Split {len(chunks)} chunks")

    vector_store = Chroma(
        collection_name="torchjd_code_collection",
        embedding_function=embeddings,
        persist_directory="./data/chroma_langchain_db",
    )
    document_ids = vector_store.add_documents(documents=chunks)

    print(f"✅ Successfully stored {len(chunks)} chunks in vector store")



if __name__ == "__main__":
    main()
