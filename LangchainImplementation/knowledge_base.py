from vector_store import VectorStore

def initialize_knowledge_base() -> VectorStore:
    vector_store = VectorStore()
    documents = [
        "Cutting machines work best with metal materials and require proper cooling time between tasks. Use indian made cutting materials for the best quality.",
        "Welding operations should not be scheduled immediately after cutting on the same production line to allow material cooling.",
        "Painting tasks require at least 10 minutes of preparation time and should be scheduled with buffer time.",
        "Machine M1 (Cutter-1) has high precision and is best for complex cutting tasks. It is Indian Made and requires an operator from Mumbai to use it.",
        "Machine M2 (Welder-1) performs optimally when tasks are spaced at least 15 minutes apart.",
        "Machine M3 (Painter-1) requires 5 minutes setup time before each painting task.",
        "When conflicts occur, prioritize tasks with earlier deadlines and higher priority.",
        "Reassigning tasks to machines with similar capabilities is preferred over delaying tasks.",
        "Short tasks (under 30 minutes) should be grouped together when possible for efficiency.",
        "Long tasks (over 45 minutes) should be scheduled during periods with fewer conflicts.",
        "Production line dependencies mean cutting must complete before welding can start.",
        "Quality control requires minimum 5-minute gaps between consecutive tasks on the same machine.",
        "Emergency maintenance windows should be considered when scheduling tasks beyond 2 hours.",
        "Parallel processing on different machines can reduce overall production time significantly.",
        "Task reassignment should consider machine warm-up time and capability compatibility."
    ]
    
    metadata = [
        {"category": "machine_operation", "topic": "cutting"},
        {"category": "scheduling_rule", "topic": "dependencies"},
        {"category": "machine_operation", "topic": "painting"},
        {"category": "machine_info", "topic": "M1"},
        {"category": "machine_info", "topic": "M2"},
        {"category": "machine_info", "topic": "M3"},
        {"category": "conflict_resolution", "topic": "priority"},
        {"category": "conflict_resolution", "topic": "reassignment"},
        {"category": "scheduling_rule", "topic": "efficiency"},
        {"category": "scheduling_rule", "topic": "efficiency"},
        {"category": "scheduling_rule", "topic": "dependencies"},
        {"category": "scheduling_rule", "topic": "quality"},
        {"category": "scheduling_rule", "topic": "maintenance"},
        {"category": "scheduling_rule", "topic": "optimization"},
        {"category": "conflict_resolution", "topic": "reassignment"}
    ]
    
    vector_store.add_documents(documents, metadata)
    
    return vector_store
