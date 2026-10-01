"""
The entire "knowledge base" for this project.

Two views of the SAME information:
  1. DOCUMENTS -> plain text chunks. These go into the VECTOR store.
  2. TRIPLES   -> (subject, relation, object, source_doc). These go into the GRAPH.

In a real project an LLM reads the documents and extracts the triples for you.
Here we write them by hand so you can see exactly what the graph contains.
"""

# 10 tiny "chunks" about a fictional company
DOCUMENTS = {
    "d1": "Priya Sharma leads the Payments team.",
    "d2": "Rahul Verma leads the Search team.",
    "d3": "Anita Rao leads the Analytics team.",
    "d4": "The Payments team stores all transactions in PostgreSQL.",
    "d5": "The Search team indexes product data in Elasticsearch.",
    "d6": "The Analytics team runs its reports on Snowflake.",
    "d7": "PostgreSQL is an open-source relational database.",
    "d8": "Elasticsearch is a search database built on Apache Lucene.",
    "d9": "Snowflake is a cloud database.",
    "d10": "The company holds an all-hands meeting every Friday.",
}

# Facts extracted from the documents: (subject, relation, object, which doc said it)
TRIPLES = [
    ("Priya Sharma", "LEADS", "Payments team", "d1"),
    ("Rahul Verma", "LEADS", "Search team", "d2"),
    ("Anita Rao", "LEADS", "Analytics team", "d3"),
    ("Payments team", "STORES_DATA_IN", "PostgreSQL", "d4"),
    ("Search team", "STORES_DATA_IN", "Elasticsearch", "d5"),
    ("Analytics team", "STORES_DATA_IN", "Snowflake", "d6"),
    ("PostgreSQL", "IS_A", "relational database", "d7"),
    ("Elasticsearch", "IS_A", "search database", "d8"),
    ("Snowflake", "IS_A", "cloud database", "d9"),
]

# A "multi-hop" question: the answer needs facts from TWO different documents.
#   d1: Priya -> Payments team
#   d4: Payments team -> PostgreSQL
# No single document mentions both "Priya" and "PostgreSQL".
QUESTION = "Which database does Priya's team use?"

# To answer correctly, the LLM must see BOTH links of the chain.
REQUIRED_DOCS = {"d1", "d4"}

