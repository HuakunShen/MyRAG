import uuid
import supabase
from typing import List, Dict, Any, Type, Optional, Union, TypeVar, Iterable
from langchain_core.documents import Document
from langchain.vectorstores.supabase import SupabaseVectorStore
from langchain.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from itertools import repeat

from modules.utils import compute_sha512

VST = TypeVar("VST", bound="VectorStore")

class CustomDocument(Document):
    source: str
    # def __init__(self, doc: Document, source: str):
    #     self.page_content = doc.page_content
    #     self.metadata = doc.metadata
    #     self.source = source
    
    @property
    def sha512(self):
        # computer sha512 hash of page_content
        compute_sha512(self.page_content + self.source)
    

class CustomSupabaseVectorStore(SupabaseVectorStore):
    @staticmethod
    def _add_vectors(
        client: supabase.client.Client,
        table_name: str,
        vectors: List[List[float]],
        documents: List[CustomDocument],
        ids: List[str],
        chunk_size: int,
    ) -> List[str]:
        """
            cls._add_vectors(client, table_name, embeddings, docs, ids, chunk_size)
        """

        rows: List[Dict[str, Any]] = [
            {
                "id": ids[idx],
                "content": documents[idx].page_content,
                "embedding": embedding,
                "metadata": documents[idx].metadata,  # type: ignore
                "source": documents[idx].source,
                "sha512": documents[idx].sha512,
            }
            for idx, embedding in enumerate(vectors)
        ]

        id_list: List[str] = []
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i : i + chunk_size]

            result = client.from_(table_name).upsert(chunk).execute()  # type: ignore

            if len(result.data) == 0:
                raise Exception("Error inserting: No rows added")

            # VectorStore.add_vectors returns ids as strings
            ids = [str(i.get("id")) for i in result.data if i.get("id")]

            id_list.extend(ids)

        return id_list
    
    @classmethod
    def from_documents(
        cls: Type[VST],
        documents: List[CustomDocument],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> VST:
        """Return VectorStore initialized from documents and embeddings."""
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        sha512s = [d.sha512 for d in documents]
        return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)

    @staticmethod
    def _texts_to_documents(
        texts: Iterable[str],
        metadatas: Optional[Iterable[Dict[Any, Any]]] = None,
    ) -> List[Document]:
        """Return list of Documents from list of texts and metadatas."""
        if metadatas is None:
            metadatas = repeat({})

        docs = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]

        return docs

    # @classmethod
    # def from_texts(
    #     cls: Type["SupabaseVectorStore"],
    #     texts: List[str],
    #     embedding: Embeddings,
    #     sha512s: List[str],
    #     metadatas: Optional[List[dict]] = None,
    #     client: Optional[supabase.client.Client] = None,
    #     table_name: Optional[str] = "documents",
    #     query_name: Union[str, None] = "match_documents",
    #     chunk_size: int = 500,
    #     ids: Optional[List[str]] = None,
    #     **kwargs: Any,
    # ) -> "SupabaseVectorStore":
    #     """Return VectorStore initialized from texts and embeddings."""
    #
    #     if not client:
    #         raise ValueError("Supabase client is required.")
    #
    #     if not table_name:
    #         raise ValueError("Supabase document table_name is required.")
    #
    #     embeddings = embedding.embed_documents(texts)
    #     ids = [str(uuid.uuid4()) for _ in texts]
    #     docs = cls._texts_to_documents(texts, metadatas)
    #     cls._add_vectors(client, table_name, embeddings, docs, ids, chunk_size)
    #
    #     return cls(
    #         client=client,
    #         embedding=embedding,
    #         table_name=table_name,
    #         query_name=query_name,
    #         chunk_size=chunk_size,
    #     )