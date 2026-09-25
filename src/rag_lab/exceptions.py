"""Domain exceptions for the RAG pipeline."""


class RagLabError(Exception):
    """Base class for all pipeline errors."""


class DataLoadError(RagLabError):
    """Raised when source documents cannot be read or parsed."""


class ChunkingError(RagLabError):
    """Raised when chunking parameters or inputs are invalid."""


class IndexingError(RagLabError):
    """Raised when embedding or vector-store operations fail."""


class RetrievalError(RagLabError):
    """Raised when a search or rerank step fails."""


class EvalError(RagLabError):
    """Raised when an evaluation label, citation, or metric input is invalid."""
