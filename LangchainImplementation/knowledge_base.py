"""
Knowledge Base for Production Scheduling RAG System

This module provides factory/production knowledge that gets embedded
into the vector store for retrieval-augmented generation (RAG).
"""

from langchain_vector_store import LangChainVectorStore


# =============================================================================
# FACTORY KNOWLEDGE DOCUMENTS
# =============================================================================

SCHEDULING_KNOWLEDGE = [
    # Machine Operations
    "Cutting machines work best with metal materials and require proper cooling time between tasks. Use indian made cutting materials for the best quality.",
    "Welding operations should not be scheduled immediately after cutting on the same production line to allow material cooling.",
    "Painting tasks require at least 10 minutes of preparation time and should be scheduled with buffer time.",
    
    # Machine-Specific Info
    "Machine M1 (Cutter-1) has high precision and is best for complex cutting tasks. It is Indian Made and requires an operator from Mumbai to use it.",
    "Machine M2 (Welder-1) performs optimally when tasks are spaced at least 15 minutes apart.",
    "Machine M3 (Painter-1) requires 5 minutes setup time before each painting task.",
    
    # Conflict Resolution
    "When conflicts occur, prioritize tasks with earlier deadlines and higher priority.",
    "Reassigning tasks to machines with similar capabilities is preferred over delaying tasks.",
    "Task reassignment should consider machine warm-up time and capability compatibility.",
    
    # Scheduling Rules
    "Short tasks (under 30 minutes) should be grouped together when possible for efficiency.",
    "Long tasks (over 45 minutes) should be scheduled during periods with fewer conflicts.",
    "Production line dependencies mean cutting must complete before welding can start.",
    "Quality control requires minimum 5-minute gaps between consecutive tasks on the same machine.",
    "Emergency maintenance windows should be considered when scheduling tasks beyond 2 hours.",
    "Parallel processing on different machines can reduce overall production time significantly.",
    
    # Maintenance Rules
    "Machines should undergo preventive maintenance every 100 operating hours.",
    "If a machine breaks down, immediately reassign pending tasks to alternative machines with matching capabilities.",
    "Maintenance windows should be scheduled during low-demand periods to minimize production impact.",
    
    # Rush Order Handling
    "Rush orders with priority 'high' should preempt normal priority tasks when possible.",
    "For rush orders, consider temporarily pausing low-priority tasks on the required machine.",
    "Rush orders should be completed within their due_date to maintain customer satisfaction.",
]

SCHEDULING_METADATA = [
    # Machine Operations
    {"category": "machine_operation", "topic": "cutting"},
    {"category": "scheduling_rule", "topic": "dependencies"},
    {"category": "machine_operation", "topic": "painting"},
    
    # Machine-Specific Info  
    {"category": "machine_info", "topic": "M1", "machine_id": "M1"},
    {"category": "machine_info", "topic": "M2", "machine_id": "M2"},
    {"category": "machine_info", "topic": "M3", "machine_id": "M3"},
    
    # Conflict Resolution
    {"category": "conflict_resolution", "topic": "priority"},
    {"category": "conflict_resolution", "topic": "reassignment"},
    {"category": "conflict_resolution", "topic": "reassignment"},
    
    # Scheduling Rules
    {"category": "scheduling_rule", "topic": "efficiency"},
    {"category": "scheduling_rule", "topic": "efficiency"},
    {"category": "scheduling_rule", "topic": "dependencies"},
    {"category": "scheduling_rule", "topic": "quality"},
    {"category": "scheduling_rule", "topic": "maintenance"},
    {"category": "scheduling_rule", "topic": "optimization"},
    
    # Maintenance Rules
    {"category": "maintenance", "topic": "preventive"},
    {"category": "maintenance", "topic": "breakdown"},
    {"category": "maintenance", "topic": "scheduling"},
    
    # Rush Order Handling
    {"category": "rush_orders", "topic": "priority"},
    {"category": "rush_orders", "topic": "preemption"},
    {"category": "rush_orders", "topic": "deadline"},
]


def initialize_knowledge_base() -> LangChainVectorStore:
    """
    Initialize the RAG knowledge base with factory scheduling knowledge.
    
    Returns:
        LangChainVectorStore: Configured vector store with embedded documents
    """
    vector_store = LangChainVectorStore()
    vector_store.add_documents(SCHEDULING_KNOWLEDGE, SCHEDULING_METADATA)
    
    return vector_store


def get_knowledge_base_with_custom_docs(additional_docs: list = None) -> LangChainVectorStore:
    """
    Initialize knowledge base with optional additional documents.
    
    Args:
        additional_docs: List of additional document strings to add
        
    Returns:
        LangChainVectorStore: Configured vector store
    """
    vector_store = initialize_knowledge_base()
    
    if additional_docs:
        vector_store.add_documents(additional_docs)
    
    return vector_store


# For backwards compatibility with old import
VectorStore = LangChainVectorStore
