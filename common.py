from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def get_vector_store() -> Chroma:

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_store = Chroma(
        collection_name="code_collection",
        embedding_function=embeddings,
    )

    return vector_store
