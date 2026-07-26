# TechGear Support Agent (RAG)

**A production-grade Retrieval-Augmented Generation engine, engineered for real-world reliability.** This repository demonstrates the system on a customer support use case, but the underlying architecture is domain-agnostic by design. Point it at any knowledge base, and it answers with the same precision, speed, and grounding, no rebuild required.


## What This System Solves

Most support and knowledge-lookup experiences are either too generic to be useful or too slow to search manually. This system solves that by retrieving the most relevant information directly from a company's own documents and generating accurate, grounded answers in real time, rather than relying on generic or hallucinated chatbot responses.

The result is a support experience that answers instantly, stays accurate to the source material, and scales to any size knowledge base without retraining a model.


---


## Core Capabilities

- **Hybrid Search Retrieval:** combines full-text keyword search with dense vector similarity search in Supabase, merged via Reciprocal Rank Fusion, for higher context precision than vector-only search.
- **Context Reranking:** re-scores retrieved document chunks using a cross-encoder model before passing only the most relevant context to the LLM.
- **Token Streaming:** delivers low-latency, word-by-word streaming responses for a natural chat experience.
- **Session Caching:** caches heavy models and active database connections to optimize response speed and resource usage.


---


## Live Demo

**[Try the Live Web App](https://your-app-name.streamlit.app)**


---


## Where This Applies

The retrieval engine underneath this demo is domain-agnostic. Swapping the source documents adapts it directly to:

- **Internal HR / IT Helpdesks:** instant answers to policy, benefits, or systems questions from company handbooks
- **Legal & Compliance Q&A:** retrieval over contracts, policies, and regulatory documents
- **Ed-Tech Study Assistants:** answering student questions from course materials, syllabi, or institutional policies
- **E-Commerce Product Support:** grounding answers in product manuals, shipping policies, and FAQs
- **Internal Company Knowledge Bases:** searchable, conversational access to internal wikis and documentation

TechGear serves as the demonstration dataset in this repository; the underlying pipeline requires no code changes to serve any of the above, only a new set of source documents.


---


## Architecture & Tech Stack

- **LLM Engine:** Groq API (Llama 3.3 70B), low-latency inference
- **Vector Database:** Supabase (`pgvector`, HNSW index, hybrid search)
- **Embeddings:** Hugging Face (Qwen3-Embedding-0.6B)
- **Reranking:** BAAI cross-encoder reranker
- **Frontend:** Streamlit
- **Language:** Python 3.10+

**Design rationale:** Supabase (Postgres + `pgvector`) was selected over a dedicated vector database because it natively supports hybrid search and deploys cleanly to production with zero extra infrastructure. Groq was chosen for the generation layer for its low-latency inference on open-source models, and Qwen3-Embedding was selected for strong retrieval performance relative to its size. Every choice was made on technical merit, not cost.


---


## Local Setup & Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ahsan03-dev/techgear-support-agent.git
   cd techgear-support-agent
   ```

2. **Create a virtual environment and install dependencies:**

   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows (CMD)
   source venv/bin/activate   # macOS/Linux
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root with:

   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Set up the database:**
   Run `schema.sql` in your Supabase SQL editor to create the table, indexes, security policies, and the hybrid search function.

5. **Ingest the knowledge base:**

   ```bash
   python ingest.py
   ```

6. **Run the app:**
   ```bash
   streamlit run app.py
   ```


---


## Adapting This to a New Domain

This repository uses TechGear as a demonstration dataset. To point the system at a different knowledge base:

1. Replace the article list in `ingest.py` with your own documents
2. Re-run `python ingest.py` to embed and store the new content
3. No changes are required to the retrieval, reranking, or generation logic

This system was engineered to a single standard: could it hold up in production, not just survive a demo. Every retrieval, ranking, and generation decision was made to meet that bar.
