from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from langchain_openai import AzureOpenAIEmbeddings
from config import settings

sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

def get_embeddings():
    """Return an OpenAIEmbeddings instance."""
    return OpenAIEmbeddings(model="text-embedding-3-large", api_key=settings.OPENAI_API_KEY)

def init_vector_store(collection_name: str = "rag_collection") -> QdrantVectorStore:
    """
    Initialize an in-memory Qdrant-backed vector store.
    All data lives in RAM (ephemeral).
    """
    # 1. Prepare embeddings
    embeddings = get_embeddings()

    # 2. Spin up an embedded, in-RAM Qdrant
    client = QdrantClient(path=":memory:")

    # 3. create the collection for fresh state
    client.create_collection(
        collection_name=collection_name,
        vectors_config={"dense": VectorParams(size=3072, distance=Distance.COSINE)},
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
        },
    )

    # 4. Wrap in LangChain vector store
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding= embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )
