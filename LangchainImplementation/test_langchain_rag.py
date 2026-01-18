"""
Test the new LangChain RAG implementation
"""

from knowledge_base import initialize_knowledge_base
from rag_chain import SchedulingRAGChain, create_rag_chain


def main():
    print("=" * 70)
    print("  TESTING LANGCHAIN RAG IMPLEMENTATION")
    print("=" * 70)
    
    # ================================================================
    # 1. Initialize Knowledge Base
    # ================================================================
    print("\n[1] Initializing LangChain Vector Store...")
    vector_store = initialize_knowledge_base()
    print(f"    ✓ Loaded {len(vector_store.documents)} documents into FAISS")
    
    # ================================================================
    # 2. Test Basic Search
    # ================================================================
    print("\n[2] Testing basic similarity search...")
    
    queries = [
        "What machine is best for cutting?",
        "How should painting tasks be scheduled?",
        "How to handle rush orders?"
    ]
    
    for query in queries:
        print(f"\n    Query: '{query}'")
        results = vector_store.search(query, top_k=2)
        for i, r in enumerate(results, 1):
            print(f"    {i}. (score: {r['score']:.3f}) {r['text'][:80]}...")
    
    # ================================================================
    # 3. Test Retriever Interface
    # ================================================================
    print("\n\n[3] Testing LangChain Retriever interface...")
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke("welding machine scheduling")
    
    print(f"    Retrieved {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        print(f"    {i}. {doc.page_content[:70]}...")
        print(f"       Metadata: {doc.metadata}")
    
    # ================================================================
    # 4. Test RAG Chain
    # ================================================================
    print("\n\n[4] Testing RAG Chain (retrieval + LLM)...")
    print("    (This may take a moment to load the LLM...)")
    
    rag_chain = create_rag_chain(vector_store)
    
    # Test getting context (no LLM)
    print("\n    Getting scheduling context...")
    context = rag_chain.get_scheduling_context("cutting machine selection")
    print(f"    Context:\n{context}")
    
    # Test machine selection
    print("\n    Testing machine selection...")
    machine = rag_chain.pick_machine(
        job_type="cutting",
        duration=45,
        available_machines=["M1", "M2", "M3"]
    )
    print(f"    Selected machine: {machine}")
    
    # ================================================================
    # 5. Test Full RAG Query
    # ================================================================
    print("\n\n[5] Testing full RAG query...")
    
    question = "What should I consider when scheduling a long cutting task?"
    print(f"    Question: {question}")
    
    result = rag_chain.query(question)
    print(f"\n    Answer: {result['result'][:200]}...")
    print(f"\n    Source documents used: {len(result['source_documents'])}")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"    {i}. {doc.page_content[:60]}...")
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("  ✓ LANGCHAIN RAG TEST COMPLETED!")
    print("=" * 70)
    print("""
    Components tested:
    - LangChainVectorStore with FAISS
    - HuggingFaceEmbeddings
    - LangChain Retriever interface
    - RetrievalQA chain
    - Custom scheduling prompts
    """)


if __name__ == "__main__":
    main()
