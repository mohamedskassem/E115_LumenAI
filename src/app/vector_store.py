import os
import logging
from typing import List, Tuple, Optional

from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.base_query_engine import BaseQueryEngine


class VectorStoreManager:
    """Manages the loading, building, and persistence of the VectorStoreIndex."""

    def __init__(self, persist_dir: str):
        """Initializes the manager with the directory for storing the index.

        Args:
            persist_dir: The directory path where the index is stored or will be stored.
        """
        self.persist_dir = persist_dir
        logging.info(
            f"VectorStoreManager initialized with persist_dir: {self.persist_dir}"
        )

    def _is_cache_valid(self) -> bool:
        """Checks if the persistence directory exists and contains key index files."""
        if not os.path.exists(self.persist_dir):
            return False

        # Check for the presence of essential index files
        # Add more checks if necessary based on the LlamaIndex version and storage backend
        docstore_path = os.path.join(self.persist_dir, "docstore.json")
        if not os.path.exists(docstore_path):
            logging.warning(
                f"Cache directory '{self.persist_dir}' exists, but key file '{docstore_path}' is missing."
            )
            return False

        logging.info(f"Found valid cache directory and key file: {docstore_path}")
        return True

    def _build_and_persist_index(
        self, schema_docs: List[Document], embed_model: BaseEmbedding
    ) -> Optional[VectorStoreIndex]:
        """Builds the vector index from documents, persists it, and returns the index.

        Args:
            schema_docs: List of LlamaIndex Document objects containing schema information.
            embed_model: The embedding model instance to use for indexing.

        Returns:
            The built VectorStoreIndex, or None if building fails.
        """
        if not embed_model:
            logging.error("Cannot build index: embed_model not provided.")
            return None
        if not schema_docs:
            logging.warning("Cannot build index: No schema documents provided.")
            return None

        try:
            logging.info(
                f"Building new vector store index with {len(schema_docs)} documents..."
            )
            index = VectorStoreIndex.from_documents(
                schema_docs, embed_model=embed_model
            )

            logging.info(f"Persisting index to: {self.persist_dir}")
            os.makedirs(self.persist_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=self.persist_dir)
            logging.info("Vector Store Index built and persisted successfully.")
            return index
        except Exception as e:
            logging.error(
                f"Error building/persisting Vector Store Index in {self.persist_dir}: {e}",
                exc_info=True,
            )
            return None

    def load_or_build_index(
        self,
        schema_docs: List[Document],
        embed_model: BaseEmbedding,
        force_rebuild: bool = False,
    ) -> Tuple[Optional[VectorStoreIndex], Optional[BaseQueryEngine]]:
        """Loads the index from cache if valid, otherwise builds and persists a new one.

        Args:
            schema_docs: List of LlamaIndex Document objects (needed for building).
            embed_model: The embedding model instance (needed for loading and building).
            force_rebuild: If True, ignore existing cache and force rebuild.

        Returns:
            A tuple containing the VectorStoreIndex and the QueryEngine,
            or (None, None) if initialization fails.
        """
        index: Optional[VectorStoreIndex] = None
        query_engine: Optional[BaseQueryEngine] = None

        # Check cache validity, but only load if force_rebuild is False
        if not force_rebuild and self._is_cache_valid():
            try:
                if not embed_model:
                    logging.error(
                        "Embed model not provided. Cannot load index from cache."
                    )
                    return None, None
                logging.info(
                    f"Loading existing vector store index from: {self.persist_dir}"
                )
                storage_context = StorageContext.from_defaults(
                    persist_dir=self.persist_dir
                )
                index = load_index_from_storage(
                    storage_context, embed_model=embed_model
                )
                logging.info("Vector Store Index loaded successfully from cache.")
            except Exception as e:
                logging.error(
                    f"Error loading index from cache {self.persist_dir}: {e}. Attempting rebuild.",
                    exc_info=True,
                )
                index = self._build_and_persist_index(schema_docs, embed_model)
        else:
            # Cache not valid, doesn't exist, or force_rebuild is True
            if force_rebuild:
                logging.info(
                    f"Force rebuild requested. Building new index in {self.persist_dir}..."
                )
            else:
                logging.info(
                    f"Vector store cache not found or invalid in {self.persist_dir}. Building new index..."
                )
            index = self._build_and_persist_index(schema_docs, embed_model)

        # If index was successfully loaded or built, create query engine
        if index:
            try:
                # Configure query engine to only retrieve context
                query_engine = index.as_query_engine(response_mode="no_text")
                logging.info("Query engine created successfully.")
            except Exception as e:
                logging.error(
                    f"Error creating query engine from index: {e}", exc_info=True
                )
                index = None  # Nullify index if query engine creation fails
                query_engine = None
        else:
            logging.error("Index initialization failed (either loading or building).")

        return index, query_engine
