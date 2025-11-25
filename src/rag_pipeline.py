from dataclasses import dataclass
from typing import List, Protocol, Optional, Dict, Any
from datetime import datetime
import json
from boto3 import client
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from logger import logger
from helpers import load_config
from log_time import log_time
import streamlit as st
import asyncio

# === Data Classes ===

@dataclass(frozen=True)
class DocumentMetadata:
    file_name: str
    file_version: str
    file_date: datetime
    section_number: str
    section_heading: str
    section_page: Optional[int] = None
    document_id: Optional[str] = None
    document_tags: Optional[List[str]] = None

    def __post_init__(self):
        if not self.file_name:
            raise ValueError("file_name cannot be empty")
        if not self.section_number:
            raise ValueError("section_number cannot be empty")

@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    content: str
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if not self.chunk_id.strip():
            raise ValueError("chunk_id cannot be empty")
        if not self.content.strip():
            raise ValueError("content cannot be empty")

@dataclass(frozen=True)
class Query:
    text: str
    embedding: List[float]

    def __post_init__(self):
        if not self.text.strip():
            raise ValueError("query text cannot be empty")
        if not self.embedding:
            raise ValueError("embedding cannot be empty")
        if not isinstance(self.embedding, list):
            raise ValueError("embedding must be a list")
        if not all(isinstance(x, float) for x in self.embedding):
            raise ValueError("embedding must contain only floats")

@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    similarity_threshold: float

    def __post_init__(self):
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    similarity_score: float

    def __post_init__(self):
        if not 0 <= self.similarity_score <= 1:
            raise ValueError("similarity_score must be between 0 and 1")

@dataclass(frozen=True)
class ProcessorConfig:
    retrieval: RetrievalConfig

# === Service Interfaces ===

class EmbeddingService(Protocol):
    def embed_text(self, text: str) -> List[float]:
        ...

class SimilarityMetric(Protocol):
    def compute(self, a: List[float], b: List[float]) -> float:
        ...

class GenerationService(Protocol):
    def generate_response(self, augmented_prompt: str) -> str:
        ...

# === For Progress ===

class ProgressCallback(Protocol):
    def update(self, message: str, **kwargs: Any) -> None: ...
    def write(self, message: str) -> None: ...
    def __call__(self, message: str, **kwargs: Any) -> None: ...

class StreamlitProgress:
    def __init__(self, status: st.status):
        self.status = status

    def __call__(self, message: str, **kwargs: Any) -> None:
        self.status.update(label=message, **kwargs)

    def write(self, message: str) -> None:
        self.status.write(message)

# For non-Streamlit environments (tests, CLI)
class NoOpProgress:
    def __call__(self, message: str, **kwargs: Any) -> None: pass
    def write(self, message: str) -> None: pass  


# === Core Corpus ===

class Corpus:
    """Repository for document chunks with duplicate prevention."""
    
    def __init__(self):
        self._chunks: List[DocumentChunk] = []
        self._chunk_ids: Dict[str, bool] = {}  # Using dict for O(1) lookup

    def _make_chunk_id(self, chunk: DocumentChunk) -> str:
        """Create a unique identifier for a chunk."""
        return f"{chunk.metadata.document_id}:{chunk.metadata.section_number}"

    def add_chunk(self, chunk: DocumentChunk) -> bool:
        """
        Add a chunk to the corpus if it doesn't exist.
        Returns True if added, False if duplicate.
        """
        chunk_id = self._make_chunk_id(chunk)
        if chunk_id in self._chunk_ids:
            logger.debug("Skipping duplicate chunk: %s", chunk_id)
            return False
        
        self._chunk_ids[chunk_id] = True
        self._chunks.append(chunk)
        return True

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Add multiple chunks to the corpus.
        Returns number of chunks actually added (excluding duplicates).
        """
        return sum(1 for chunk in chunks if self.add_chunk(chunk))

    def get_all_chunks(self) -> List[DocumentChunk]:
        """Get all chunks in the corpus."""
        return self._chunks.copy()  # Return a copy to prevent modification

    def clear(self) -> None:
        """Clear all chunks from the corpus."""
        self._chunks.clear()
        self._chunk_ids.clear()

    def __len__(self) -> int:
        return len(self._chunks)

# === Similarity Implementation ===

class CosineSimilarity:
    """Compute cosine similarity between two vectors."""
    
    def compute(self, a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            raise ValueError("Vectors must have same dimension")
        
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            raise ValueError("Zero vectors are not allowed")
            
        return dot / (norm_a * norm_b)

# === Retrieval ===

class RetrievalService:
    """Service for retrieving relevant chunks based on query similarity."""
    
    def __init__(self, 
                 vector_svc: client,
                 s3_client: client, 
                 reranker_model_name: str = None):
            
        self.vector_client = vector_svc
        self.s3_client = s3_client
        self.raw_bucket = load_config("AWS_S3_BUCKET")
        self.chunks_prefix = "chunks/"
        
        logger.info("Initialized RetrievalService")
        
        # initialize BGE reranker if requested
        if reranker_model_name:
            # Use local_files_only since model is pre-downloaded in Docker image
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                reranker_model_name,
                local_files_only=True
            )
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                reranker_model_name,
                local_files_only=True
            )
            logger.info(f"Loaded BGE re-ranker model from cache: {reranker_model_name}")
        else:
            self.reranker_tokenizer = None
            self.reranker_model = None

    @log_time("rerank_with_bge")
    async def rerank_with_bge(
        self,
        query_text: str,
        retrieved_chunks: List[RetrievedChunk],
        top_n: int = 1
    ) -> List[RetrievedChunk]:
        """
        Re-rank retrieved chunks with BGE cross-encoder
        """
        if not self.reranker_model:
            logger.warning("BGE re-ranker is not configured, skipping reranking.")
            return retrieved_chunks
        
        pairs = [(query_text, rc.chunk.content) for rc in retrieved_chunks]
        
        def _run_reranker():
            inputs = self.reranker_tokenizer(
                [q for q, c in pairs],
                [c for q, c in pairs],
                return_tensors="pt",
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.reranker_model(**inputs)
                scores = outputs.logits.squeeze(-1)
            return scores

        scores = await asyncio.to_thread(_run_reranker)        
        # rank the retrieved chunks by BGE re-ranker scores
        sorted_indices = torch.topk(scores, k=min(top_n, len(retrieved_chunks))).indices.tolist()
        
        sorted_indices = [x for x in sorted_indices if scores[x].item() > scores[sorted_indices[0]].item()*0.5]
        
        reranked_chunks = [retrieved_chunks[i] for i in sorted_indices]
        
        for idx,rc in enumerate(reranked_chunks):
            logger.info("Re-ranked chunk from section %s with BGE score %.3f", 
                        rc.chunk.metadata.section_number, scores[sorted_indices[idx]].item())
        
        logger.info("Re-ranked chunks with BGE cross-encoder.")
        return reranked_chunks

    @log_time("retrieve_similar_chunks")
    async def retrieve_similar_chunks(
        self, query: Query, config: RetrievalConfig
    ) -> List[RetrievedChunk]:
        """
        Retrieve chunks similar to the query based on the config.
        Returns list of chunks sorted by similarity score.
        """
        resp = await asyncio.to_thread(
            self.vector_client.query_vectors,
            vectorBucketName=load_config("VECTOR_S3_BUCKET"),
            indexName=load_config("VECTOR_INDEX"),
            queryVector={'float32':query.embedding},
            topK=config.top_k,
            returnMetadata=False,
            returnDistance=True,
        )
        hits = resp.get("vectors", []) 
        
        logger.info("Retrieved %d chunks from S3 vectors", len(hits))
        scores = []
        for hit in hits:
            scores.append(round(hit.get("distance"),2))
        logger.info(f"  with scores: {scores}")
        
        # 2) Group hits by document hash
        docs_to_ids = {}
        for hit in hits:
            cid = hit["key"]
            doc_hash, _ = cid.split("_", 1)
            docs_to_ids.setdefault(doc_hash, []).append(cid)

        # 3) Load JSON files once per doc_hash
        chunks_store = {}
        async def fetch_doc(doc_hash: str, id_list: list):
            key = f"{self.chunks_prefix}{doc_hash}.json"
            obj = await asyncio.to_thread(
                self.s3_client.get_object,
                Bucket=self.raw_bucket, Key=key)
            raw = obj["Body"].read().decode("utf-8")
            entries = json.loads(raw)
            for entry in entries:
                if entry["chunk_id"] in id_list:
                    chunks_store[entry["chunk_id"]] = entry

        await asyncio.gather(*[
            fetch_doc(doc_hash, ids)
            for doc_hash, ids in docs_to_ids.items()
        ])

        # 4) Build RetrievedChunk list
        results = []
        for hit in hits:
            cid = hit["key"]
            data = chunks_store.get(cid)
            if not data:
                continue
            # Reconstruct DocumentMetadata + DocumentChunk
            meta = data["metadata"]
            metadata = DocumentMetadata(
                file_name=meta["file_name"],
                file_version=meta["file_version"],
                file_date=datetime.fromisoformat(meta["file_date"]),
                section_number=meta["section_number"],
                section_heading=meta["section_heading"],
                section_page=meta["section_page"],
                document_id=meta["document_id"]
            )
            chunk = DocumentChunk(
                chunk_id=cid,
                content=data["content"],
                metadata=metadata
            )
            # Distance → similarity score (optionally invert)
            score = hit.get("distance")
            results.append(RetrievedChunk(chunk=chunk, similarity_score=score))

        # 5) Filter, sort, and rerank
        #results = [r for r in results if r.similarity_score >= config.similarity_threshold]
        results.sort(key=lambda r: r.similarity_score, reverse=True)

        logger.info("Retrieved %d chunks above threshold %.2f", len(results), config.similarity_threshold)
        for r in results:
            logger.info("Chunk %s score %.3f", r.chunk.chunk_id, r.similarity_score)

        if self.reranker_model and len(results) > 1:
            logger.info(f"Reranking {len(results)} chunks using {self.reranker_model}..")
            results = await self.rerank_with_bge(query.text, results)

        return results

# === Prompt Augmentation ===

class PromptAugmenter:
    """Service for augmenting queries with retrieved context."""
    
    def __init__(self, prompt_template_path: Optional[str] = None):
        """
        Initialize with optional prompt template path.
        If not provided, uses default template.
        """
        self.prompt_template = self._load_template(prompt_template_path)

    def _load_template(self, template_path: Optional[str]) -> str:
        """Load prompt template from file or use default."""
        if template_path:
            with open(template_path, 'r') as f:
                return f.read()
        return "{user_query}\n\nContext:\n{retrieved_chunks_text}"

    def augment_query(self, query: Query, retrieved_chunks: List[RetrievedChunk]) -> str:
        """
        Augment query with retrieved chunks using template.
        """
        logger.info("Augmenting query with %d retrieved chunks", len(retrieved_chunks))
        
        retrieved_chunk_text = ""
        for rc in retrieved_chunks:
            md = rc.chunk.metadata
            context_line = f"[Section {md.section_number} of {md.file_name}]: {rc.chunk.content}"
            retrieved_chunk_text += context_line + "\n"
            
        prompt = self.prompt_template.format(
            user_query=query.text,
            retrieved_chunks_text=retrieved_chunk_text
        )
        
        estimated_tokens = len(prompt) // 4
        logger.debug("Generated augmented prompt with ~%d tokens", estimated_tokens)
        return prompt

# === Query Processor ===

class QueryProcessor:
    """Main service for processing queries using RAG pipeline."""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        retrieval_service: RetrievalService,
        prompt_augmenter: PromptAugmenter,
        generation_service: GenerationService,
        config: ProcessorConfig
    ):
        self.embedding_service = embedding_service
        self.retrieval_service = retrieval_service
        self.prompt_augmenter = prompt_augmenter
        self.generation_service = generation_service
        self.config = config
        logger.info("Initialized QueryProcessor with config: %s", config)

    async def pre_gen_process(
        self, 
        query_text: str,
        progress: Optional[ProgressCallback] = None,
        ) -> str:
        """
        Process a query through the RAG pipeline until the generation.
        Returns augmented response.
        """
        progress = progress or NoOpProgress()

        if not query_text.strip():
            raise ValueError("Query text cannot be empty")

        logger.info("Processing query: %s", query_text)
        progress("Embedding your question...")
        query_embedding = await self.embedding_service.embed_text(query_text)

        progress("Searching the vector database for relevant chunks...")
        query = Query(text=query_text, embedding=query_embedding)

        retrieved_chunks = await self.retrieval_service.retrieve_similar_chunks(
            query, self.config.retrieval
        )
        if retrieved_chunks:
            progress("Re-ranking and filtering the best matches...")
            augmented_prompt = self.prompt_augmenter.augment_query(query, retrieved_chunks)
        else:
            augmented_prompt = query_text
        logger.info("Augmented prompt: %s", augmented_prompt)
        return augmented_prompt

    async def process_query(
        self, 
        query_text: str,
        progress: Optional[ProgressCallback] = None,
        ) -> str:
        """
        Process a query through the RAG pipeline after retrieval.
        Returns generated response.
        """
        progress = progress or NoOpProgress()

        augmented_prompt = await self.pre_gen_process(query_text)
        if augmented_prompt == query_text:
            progress("No relevant documents found → generating a fun fact...")
            response = "You query is unrelated to the subject in corpus. "
            response += "😊 Would you like to make another query?\n\n\n\n"
            response += "💡 Did You Know? 💡\n\n"
            didyouknowprompt = f"Generate a random and fun and insightful did-you-know factoid about CIS Benchmarks in less than 30 words. If possible, make it relevant to the user query here: ***{query_text}***; it can even be technical if relevant to user query. Just give the factoid, nothing else. (no preamble etc)"
            response += await self.generation_service.generate_response(didyouknowprompt) 
        else:
            progress("Generating final answer with retrieved context...")
            response = await self.generation_service.generate_response(augmented_prompt)
        logger.info("Query processing completed")
        logger.info("Response: %s", response)
        progress("Finalizing response...", state="complete")
        return response
