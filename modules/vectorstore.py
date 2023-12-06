import uuid
import supabase
from supabase.client import Client
from typing import List, Dict, Any, Type, Optional, Union, TypeVar, Iterable, Tuple
from langchain_core.documents import Document
from langchain.vectorstores.supabase import SupabaseVectorStore
from langchain.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from itertools import repeat

from modules.utils import compute_sha512

VST = TypeVar("VST", bound="VectorStore")


class CustomDocument(Document):
    source_id: int
    source: str
    collection: str
    # def __init__(self, doc: Document, source: str):
    #     self.page_content = doc.page_content
    #     self.metadata = doc.metadata
    #     self.source = source
    
    @property
    def sha512(self) -> str:
        # computer sha512 hash of page_content
        return compute_sha512(self.page_content)
    

class CustomSupabaseVectorStore(SupabaseVectorStore):
    @classmethod
    def from_docs(cls, client: Client, table_name: str, documents: List[CustomDocument], embedding: Embeddings, collection: str, chunk_size: int = 500) -> VST:
        if not client:
            raise ValueError("Supabase client is required.")

        if not table_name:
            raise ValueError("Supabase document table_name is required.")
        texts = [d.page_content for d in documents]
        embeddings = embedding.embed_documents(texts)
        ids = [str(uuid.uuid4()) for _ in texts]
        cls._add_vectors(client, table_name, embeddings, documents, ids, chunk_size)

        return cls(
            client=client,
            embedding=embedding,
            table_name=table_name,
            query_name="match_documents",
            chunk_size=chunk_size,
        )

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
                "source_id": documents[idx].source_id,
                "sha512": documents[idx].sha512,
                "collection": documents[idx].collection,
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
            CustomDocument(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]

        return docs

    def match_args(
        self, query: List[float], filter: Optional[Dict[str, Any]], collection: str
    ) -> Dict[str, Any]:
        ret: Dict[str, Any] = dict(query_embedding=query, collection_name=collection)
        if filter:
            ret["filter"] = filter
        return ret


    def similarity_search_by_vector_with_relevance_scores(
        self,
        query: List[float],
        k: int,
        collection: str,
        filter: Optional[Dict[str, Any]] = None,
        postgrest_filter: Optional[str] = None,
    ) -> List[Tuple[CustomDocument, float]]:
        match_documents_params = self.match_args(query, filter, collection)
        query_builder = self._client.rpc(self.query_name, match_documents_params)
        # print(match_documents_params)
        if postgrest_filter:
            query_builder.params = query_builder.params.set(
                "and", f"({postgrest_filter})"
            )

        query_builder.params = query_builder.params.set("limit", k)

        res = query_builder.execute()

        match_result = [
            (
                CustomDocument(
                    metadata=search.get("metadata", {}),  # type: ignore
                    page_content=search.get("content", ""),
                    collection=search.get("collection"),
                    source_id=search.get("source_id"),
                    source=search.get("source")
        ),
                search.get("similarity", 0.0),
            )
            for search in res.data
            if search.get("content")
        ]

        return match_result
    

    def similarity_search(
        self,
        query: str,
        collection: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[CustomDocument]:
        vector = self._embedding.embed_query(query)
        return self.similarity_search_by_vector(vector, k=k, filter=filter, collection=collection, **kwargs)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        collection: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[CustomDocument]:
        result = self.similarity_search_by_vector_with_relevance_scores(
            embedding, k=k, filter=filter, collection=collection, **kwargs
        )

        documents = [doc for doc, _ in result]

        return documents