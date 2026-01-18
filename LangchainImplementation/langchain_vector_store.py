"""
LangChain RAG Vector Store

This module provides a proper LangChain-based RAG implementation using:
- FAISS for vector storage
- HuggingFaceEmbeddings for embeddings
- LangChain Retriever interface
"""

from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


class LangChainVectorStore:
    """
    LangChain-based vector store using FAISS.
    Provides proper RAG integration with retrievers and chains.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store with HuggingFace embeddings.
        
        Args:
            model_name: HuggingFace model for embeddings
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore: Optional[FAISS] = None
        self.documents: List[Document] = []
    
    def add_documents(self, texts: List[str], metadata: List[Dict[str, Any]] = None):
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text strings to add
            metadata: Optional list of metadata dicts for each text
        """
        if metadata is None:
            metadata = [{} for _ in texts]
        
        # Create LangChain Document objects
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadata)
        ]
        self.documents.extend(docs)
        
        # Create or update FAISS index
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vectorstore.add_documents(docs)
    
    def add_texts_with_splitting(self, texts: List[str], chunk_size: int = 500, 
                                  chunk_overlap: int = 50):
        """
        Add documents with text splitting for better retrieval.
        
        Args:
            texts: List of text strings (can be long)
            chunk_size: Max size of each chunk
            chunk_overlap: Overlap between chunks
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        docs = []
        for i, text in enumerate(texts):
            chunks = splitter.split_text(text)
            for j, chunk in enumerate(chunks):
                docs.append(Document(
                    page_content=chunk,
                    metadata={"source_idx": i, "chunk_idx": j}
                ))
        
        self.documents.extend(docs)
        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vectorstore.add_documents(docs)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of dicts with 'text', 'score', and 'metadata'
        """
        if not self.vectorstore:
            return []
        
        # Use similarity_search_with_score for relevance scores
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        return [
            {
                "text": doc.page_content,
                "score": float(1 - score),  # Convert distance to similarity
                "metadata": doc.metadata
            }
            for doc, score in results
        ]
    
    def as_retriever(self, search_kwargs: Dict = None):
        """
        Get a LangChain Retriever interface.
        
        Args:
            search_kwargs: Retriever config (e.g., {"k": 3})
            
        Returns:
            LangChain Retriever
        """
        if not self.vectorstore:
            raise ValueError("No documents added to vector store yet")
        
        if search_kwargs is None:
            search_kwargs = {"k": 3}
        
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def get_relevant_documents(self, query: str, k: int = 3) -> List[Document]:
        """
        Get relevant documents using the retriever interface.
        
        Args:
            query: Search query
            k: Number of documents
            
        Returns:
            List of LangChain Document objects
        """
        retriever = self.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)
    
    def save(self, folder_path: str):
        """Save the vector store to disk."""
        if self.vectorstore:
            self.vectorstore.save_local(folder_path)
    
    def load(self, folder_path: str):
        """Load the vector store from disk."""
        if os.path.exists(folder_path):
            self.vectorstore = FAISS.load_local(
                folder_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )


# Keep old VectorStore for backwards compatibility
class VectorStore(LangChainVectorStore):
    """Alias for backwards compatibility"""
    pass
