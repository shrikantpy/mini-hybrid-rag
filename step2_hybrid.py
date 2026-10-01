"""
STEP 2 - Hybrid graph + vector RAG.  Run:  python step2_hybrid.py

The flow (4 steps):
  1. VECTOR SEARCH   find the best entry-point chunk(s) for the question
  2. ENTITY LINKING  which graph nodes are mentioned in those chunks?
  3. GRAPH EXPAND    hop along edges to reach connected facts
  4. BUILD CONTEXT   entry chunks + the chunks behind the edges we walked
"""
from data import DOCUMENTS, TRIPLES, QUESTION, REQUIRED_DOCS
from vector_store import VectorStore
from graph_store import GraphStore

TOP_K = 1   # only need ONE good entry point - the graph does the rest
HOPS = 2    # Priya -> Payments team -> PostgreSQL -> relational database

vectors = VectorStore(DOCUMENTS)
graph = GraphStore(TRIPLES)

print(f"QUESTION: {QUESTION}\n")

# 1. VECTOR SEARCH - "where do I start?"
hits = vectors.search(QUESTION, top_k=TOP_K)
print("1) VECTOR SEARCH (entry point)")
for doc_id, score in hits:
    print(f"   [{doc_id}] score={score:.3f}  {DOCUMENTS[doc_id]}")

# 2. ENTITY LINKING - "which graph nodes are in that chunk?"
entry_text = " ".join(DOCUMENTS[doc_id] for doc_id, _ in hits)
start_nodes = graph.find_entities(entry_text)
print(f"\n2) ENTITY LINKING -> start nodes: {start_nodes}")

# 3. GRAPH EXPANSION - "what is connected to those nodes?"
edges = graph.expand(start_nodes, hops=HOPS)
print(f"\n3) GRAPH EXPANSION ({HOPS} hops) -> facts walked:")
for subject, relation, obj, source in edges:
    print(f"   ({subject}) -[{relation}]-> ({obj})   from {source}")

# 4. BUILD CONTEXT - entry chunks + every chunk that backs a walked edge
doc_ids = [doc_id for doc_id, _ in hits]
for *_, source in edges:
    if source not in doc_ids:
        doc_ids.append(source)

context = "\n".join(f"[{d}] {DOCUMENTS[d]}" for d in doc_ids)
print("\n4) CONTEXT that would be sent to the LLM:")
print("-" * 60)
print(context)
print("-" * 60)

missing = REQUIRED_DOCS - set(doc_ids)
print("\nRESULT:", f"FAIL - missing {sorted(missing)}" if missing
      else "PASS - the full chain Priya -> Payments team -> PostgreSQL is in context")
print("Expected answer: Priya leads the Payments team, which stores its data in "
      "PostgreSQL (a relational database).")

# Optional: really ask an LLM. Needs `pip install anthropic` + ANTHROPIC_API_KEY.
try:
    import os
    import anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        client = anthropic.Anthropic()
        reply = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=200,
            messages=[{"role": "user", "content":
                       f"Answer using ONLY this context.\n\nContext:\n{context}\n\n"
                       f"Question: {QUESTION}"}],
        )
        print("\nLLM ANSWER:", reply.content[0].text)
except ImportError:
    pass  # LLM step is optional - the retrieval above is the lesson
