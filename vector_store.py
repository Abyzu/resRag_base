import os
import faiss
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext, load_index_from_storage


class FaissStore:
    def __init__(self, directory_path, dim=768):
        self.dim = dim
        self.dir = directory_path
        self.index = self.initialize_index()

    def initialize_index(self):
        if os.path.exists("index-storage-context") and os.listdir(
            "index-storage-context"
        ):
            print("Loading from storage-default")
            vector_store = FaissVectorStore.from_persist_dir("index-storage-context")
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir="index-storage-context"
            )
            index = load_index_from_storage(storage_context)
            return index
        else:
            documents = SimpleDirectoryReader(self.dir).load_data()
            faiss_index = faiss.IndexFlatL2(self.dim)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(
                documents=documents, storage_context=storage_context
            )
            index.storage_context.persist(persist_dir="index-storage-context")
            return index

    def get_query_engine(self):
        return self.index.as_query_engine()
