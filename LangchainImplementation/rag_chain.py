"""
LangChain RAG Chain for Production Scheduling

This module provides proper RAG chains that combine:
- Document retrieval from vector store
- LLM generation with retrieved context
- Uses new LCEL (LangChain Expression Language) patterns
"""

from typing import Any, Dict, List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


# =============================================================================
# PROMPT TEMPLATES FOR SCHEDULING DECISIONS
# =============================================================================

SCHEDULING_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an intelligent production scheduler. Use the following knowledge to make scheduling decisions.

Relevant Knowledge:
{context}

Question: {question}

Provide a clear, actionable answer based on the knowledge above. Be concise.

Answer:"""
)

MACHINE_SELECTION_PROMPT = PromptTemplate(
    input_variables=["context", "job_type", "duration", "machines"],
    template="""You are a production scheduler. Select the best machine for this job.

Job Details:
- Type: {job_type}
- Duration: {duration} minutes

Available Machines: {machines}

Relevant Knowledge:
{context}

Based on the knowledge above, which machine should handle this job? 
Return ONLY the machine name (e.g., "M1" or "Cutter-1"), nothing else.

Best Machine:"""
)


def format_docs(docs):
    """Format retrieved documents into a context string."""
    return "\n".join([f"- {doc.page_content}" for doc in docs])


class SchedulingRAGChain:
    """
    RAG Chain specifically designed for production scheduling decisions.
    Uses LangChain Expression Language (LCEL) for modern chain composition.
    """
    
    def __init__(self, vector_store, llm=None):
        """
        Initialize the RAG chain.
        
        Args:
            vector_store: LangChainVectorStore instance
            llm: Optional LLM (defaults to GPT-2 pipeline)
        """
        self.vector_store = vector_store
        
        # Setup LLM
        if llm is None:
            print("    Loading GPT-2 as default LLM (this is a placeholder)...")
            hf_pipeline = pipeline(
                "text-generation",
                model="gpt2",
                max_new_tokens=100,
                device=-1,  # CPU
                pad_token_id=50256  # Suppress warning
            )
            self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        else:
            self.llm = llm
        
        # Create retriever
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Create RAG chain using LCEL
        self.rag_chain = (
            {
                "context": self.retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
            | SCHEDULING_PROMPT
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a scheduling question.
        
        Args:
            question: The scheduling question
            
        Returns:
            Dict with 'result' (answer) and 'source_documents'
        """
        # Get source documents
        source_docs = self.retriever.invoke(question)
        
        # Get answer from chain
        result = self.rag_chain.invoke(question)
        
        return {
            "result": result,
            "source_documents": source_docs
        }
    
    def get_scheduling_context(self, query: str, k: int = 3) -> str:
        """
        Get relevant scheduling context without LLM generation.
        Useful when you want to inject context into your own prompts.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            Formatted context string
        """
        docs = self.retriever.invoke(query)
        
        context_parts = []
        for i, doc in enumerate(docs[:k], 1):
            context_parts.append(f"{i}. {doc.page_content}")
        
        return "\n".join(context_parts)
    
    def pick_machine(self, job_type: str, duration: int, available_machines: List[str]) -> str:
        """
        Use RAG to pick the best machine for a job.
        
        Args:
            job_type: Type of job (cutting, welding, painting, etc.)
            duration: Job duration in minutes
            available_machines: List of available machine IDs
            
        Returns:
            Selected machine ID
        """
        # Get context about this job type
        context_docs = self.retriever.invoke(f"{job_type} machine scheduling")
        context = format_docs(context_docs)
        
        # Build machine selection chain
        machine_chain = (
            MACHINE_SELECTION_PROMPT
            | self.llm
            | StrOutputParser()
        )
        
        result = machine_chain.invoke({
            "context": context,
            "job_type": job_type,
            "duration": duration,
            "machines": ", ".join(available_machines)
        })
        
        # Extract just the machine name
        result = result.strip()
        
        # If result contains one of our machines, return it
        for machine in available_machines:
            if machine.lower() in result.lower():
                return machine
        
        # Default fallback
        return available_machines[0] if available_machines else "M1"


def create_rag_chain(vector_store, llm=None) -> SchedulingRAGChain:
    """
    Factory function to create a SchedulingRAGChain.
    
    Args:
        vector_store: LangChainVectorStore instance
        llm: Optional LLM
        
    Returns:
        SchedulingRAGChain instance
    """
    return SchedulingRAGChain(vector_store, llm)
