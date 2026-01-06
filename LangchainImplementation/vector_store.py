from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the vector store with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = []
        self.metadata = []
    
    def add_documents(self, texts: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents to the vector store."""
        if metadata is None:
            metadata = [{} for _ in texts]
        
        embeddings = self.model.encode(texts)
        
        self.documents.extend(texts)
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadata)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity."""
        if not self.documents:
            return []
        
        query_embedding = self.model.encode([query])[0]
        
        # Calculate cosine similarity - basically a . b / a x b 
        similarities = []
        for emb in self.embeddings:
            similarity = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            similarities.append(similarity)
        
        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": self.documents[idx],
                "score": float(similarities[idx]),
                "metadata": self.metadata[idx]
            })
        
        return results
    
    def save(self, filepath: str):
        """Save the vector store to disk."""
        data = {
            "documents": self.documents,
            "embeddings": [emb.tolist() for emb in self.embeddings],
            "metadata": self.metadata
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath: str):
        """Load the vector store from disk."""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.documents = data["documents"]
        self.embeddings = [np.array(emb) for emb in data["embeddings"]]
        self.metadata = data["metadata"]
