"""
STEP 1 - Plain vector RAG.  Run:  python step1_vector_only.py

Watch it FAIL on a multi-hop question. That failure is the whole reason
hybrid graph + vector RAG exists.
"""
from data import DOCUMENTS, QUESTION, REQUIRED_DOCS
from vector_store import VectorStore

TOP_K = 2

store = VectorStore(DOCUMENTS)

print(f"QUESTION: {QUESTION}\n")
print("Similarity of EVERY chunk to the question:")
for doc_id, score in store.search(QUESTION, top_k=len(DOCUMENTS)):
    marker = "  <-- needed for the answer" if doc_id in REQUIRED_DOCS else ""
    print(f"  [{doc_id:>3}] {score:.3f}  {DOCUMENTS[doc_id]}{marker}")

results = store.search(QUESTION, top_k=TOP_K)
retrieved = [doc_id for doc_id, _ in results]
context = "\n".join(f"[{d}] {DOCUMENTS[d]}" for d in retrieved)

print(f"\nTop {TOP_K} retrieved -> CONTEXT sent to the LLM:")
print("-" * 60)
print(context)
print("-" * 60)

missing = REQUIRED_DOCS - set(retrieved)
if missing:
    print(f"\nRESULT: FAIL - missing chunk(s): {sorted(missing)}")
    print("""
WHY IT FAILED
  The question shares words with:
    - the chunk about Priya            ('Priya')
    - chunks about *some* database     ('database')
  So the LLM sees "Priya leads Payments" + "Snowflake is a database"
  and will likely answer Snowflake - confidently WRONG.

  The chunk that actually links them,
    [d4] "The Payments team stores all transactions in PostgreSQL."
  contains neither 'Priya' nor 'database', so it ranks near the BOTTOM.
  (Raising TOP_K does not fix this - you just pull in more noise.)

  Similarity != connection.   Next: python step2_hybrid.py""")
else:
    print("\nRESULT: PASS - all required chunks retrieved")
