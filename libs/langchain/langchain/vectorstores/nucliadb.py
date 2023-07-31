"""Wrapper around NucliaDB vector database."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance

logger = logging.getLogger(__name__)


class NucliaDB(VectorStore):
    """Wrapper around NucliaDB vector database.

    To use, you should have the ``nucliadb_sdk`` python package installed.

    Example:
        .. code-block:: python

            from langchain.vectorstores import NucliaDB
            from langchain.embeddings.openai import OpenAIEmbeddings
            import nucliadb_sdk

            # The environment should be the one specified next to the API key
            index = nucliadb_sdk.get_or_create("langchain-demo", nucliadb_base_url="...")
            embeddings = OpenAIEmbeddings()
            vectorstore = NucliaDB(index, embeddings)


            # Another way to create an index
            from nucliadb_sdk import *
            sdk = NucliaSDK(url="http://0.0.0.0:8080/api", region=Region.ON_PREM)
            kb = sdk.create_knowledge_box(slug="my_coffee_kb")



    """

    def __init__(
        self,
        client: NucliaSDK,
        embedding: Embeddings,
        index_name: str,
        vectorset: Optional[str] = None,
    ):
        """Initialize NucliaDB KnoledgeBase client"""
        try:
            import nucliadb_sdk
        except ImportError:
            raise ValueError(
                "Could not import NucliaDB SDK python package. Please install it with `pip install nucliadb_sdk`."
            )

        if not isinstance(client, nucliadb_sdk.NucliaSDK):
            raise ValueError(f"client should be an instance of nucliadb_sdk.KnowledgeBox, got {type(index)}")

        self._client = client
        self._index_name = index_name
        self._embedding = embedding
        vectorset = vectorset or "default-vectorset"
        self._vectorset = vectorset

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        vectorset: Optional[str] = None,
        batch_size: int = 32,  # TODO: make it in batches
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            vectorset: Optional name of the vectorset to use in the vector database.
            batch_size: int with the batch size for embedding the texts
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if vectorset is None:
            vectorset = self._vectorset
        ids = []
        for text in texts:
            embedding = self._embedding.embed_query(text)
            ids.append(self._index.upload(text=text, vectors={vectorset: embedding}))

    def similarity_search_with_score(self, query: str, k: int = 4, vectorset: Optional[str] = None) -> List[Document]:
        """Return docs most similar to query."""
        from nucliadb_models.search import ResourceProperties, SearchOptions
        vectorset = vectorset or self._vectorset
        query_vector = self._embedding.embed_query(query)
        docs = []
        kb = self._client.get_knowledge_box_by_slug(slug=self._index_name)
        results = self._client.search(kbid=kb.uuid, vector=query_vector, query=query, vectorset=vectorset, min_score=0.4, features=[SearchOptions.VECTOR], show=[ResourceProperties.BASIC, ResourceProperties.VALUES], sort={"field": "score", "limit": k})
        for i in results.sentences.results:
            content = results.resources[i.rid].data.texts["text"].value.body
            score = i.score
            # TODO: Add k, text, vectorset and min_score from kwargs
            docs.append((Document(page_content=content), score))
        return docs

    def similarity_search(self, query: str, k: int = 4, vectorset: Optional[str] = None, **kwargs: Any) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(query, k, vectorset)
        return [doc for doc, score in docs_and_scores]

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """

        if ids is None:
            return None
        for id in ids:
            del self._index[id]
        return None


    @classmethod
    def from_texts(
        cls: Type[NucliaDB],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        index_name: str = "langchain-index",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        vectorset: Optional[str] = None,
        **kwargs: Any,
    ) -> NucliaDB:
        """Return VectorStore initialized from texts and embeddings."""
        try:
            import nucliadb_sdk
        except ImportError:
            raise ValueError(
                "Could not import NucliaDB SDK python package. Please install it with `pip install nucliadb_sdk`."
            )
        # index = nucliadb_sdk.get_or_create(index_name, nucliadb_base_url="...")  # Todo: use the NucliaSDK->v2.sdk.NucliaDB for both cloud and local URLS. This is the older version
        # client = NucliaSDK(api_key=api_key)  # Todo: implement cloud database client
        # client = NucliaSDK(api=api_key)
        client = nucliadb_sdk.NucliaSDK(
            api_key="api-key", url="http://0.0.0.0:8080/api", region=nucliadb_sdk.Region.ON_PREM
        )
        try:
            kb = client.get_knowledge_box_by_slug(slug=index_name)
        except nucliadb_sdk.exceptions.NotFoundError:
            kb = client.create_knowledge_box(slug=index_name)
        vectorset = vectorset or "default-vectorset"
        for text in texts:
            embeds = embedding.embed_documents([text])
            # TODO: get embeddings of chunks and batch upload them
            client.create_resource(
                kbid=kb.uuid,
                texts={"text": {"body": text}},
                uservectors=[cls._user_vector(vectorset, embeds[0])],
            )

        return cls(client, embedding, index_name, vectorset)

    @staticmethod
    def _user_vector(vectorset, vectors):
            return {"field": {"field": "text","field_type": "text",},
            "vectors": {vectorset: {"vectors": {"vector": vectors}}}}
