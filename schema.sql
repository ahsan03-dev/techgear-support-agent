-- TechGear Support Agent — Database Schema
-- Run this in the Supabase SQL editor before running ingest.py

-- 1. Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create support articles table
CREATE TABLE IF NOT EXISTS public.support_articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    fts TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))
    ) STORED,
    embedding VECTOR(1024), -- Dimension size for Qwen3-Embedding-0.6B
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 3. Enable Row Level Security (RLS)
ALTER TABLE public.support_articles ENABLE ROW LEVEL SECURITY;

-- 4. Policy: allow public/anon read access for RAG queries
CREATE POLICY "Allow public read access for support articles"
ON public.support_articles
FOR SELECT
TO anon, authenticated
USING (true);

-- 5. Policy: allow public/anon inserts (used during ingestion)
CREATE POLICY "Allow public insert for support articles"
ON public.support_articles FOR INSERT
TO anon, authenticated
WITH CHECK (true);

-- 6. Policy: allow full write access via service_role
CREATE POLICY "Allow write access for service role"
ON public.support_articles
FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

-- 7. Full-text search (keyword) index
CREATE INDEX IF NOT EXISTS support_articles_fts_idx
ON public.support_articles USING gin(fts);

-- 8. Vector index (HNSW for high-performance similarity search)
CREATE INDEX IF NOT EXISTS support_articles_embedding_hnsw_idx
ON public.support_articles USING hnsw (embedding vector_cosine_ops);

-- 9. Hybrid search function (vector + keyword via Reciprocal Rank Fusion)
CREATE OR REPLACE FUNCTION match_support_articles_hybrid(
    query_text TEXT,
    query_embedding VECTOR(1024),
    match_count INT DEFAULT 10,
    rrf_k INT DEFAULT 60
)
RETURNS TABLE (
    id UUID,
    category TEXT,
    title TEXT,
    content TEXT,
    score FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH vector_matches AS (
        SELECT
            sa.id,
            ROW_NUMBER() OVER (ORDER BY sa.embedding <=> query_embedding) AS rank
        FROM public.support_articles sa
        ORDER BY sa.embedding <=> query_embedding
        LIMIT match_count * 2
    ),
    keyword_matches AS (
        SELECT
            sa.id,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(sa.fts, websearch_to_tsquery('english', query_text)) DESC) AS rank
        FROM public.support_articles sa
        WHERE sa.fts @@ websearch_to_tsquery('english', query_text)
        ORDER BY ts_rank_cd(sa.fts, websearch_to_tsquery('english', query_text)) DESC
        LIMIT match_count * 2
    )
    SELECT
        sa.id,
        sa.category,
        sa.title,
        sa.content,
        (COALESCE(1.0 / (rrf_k + vm.rank), 0.0) + COALESCE(1.0 / (rrf_k + km.rank), 0.0))::FLOAT AS score
    FROM public.support_articles sa
    LEFT JOIN vector_matches vm ON sa.id = vm.id
    LEFT JOIN keyword_matches km ON sa.id = km.id
    WHERE vm.id IS NOT NULL OR km.id IS NOT NULL
    ORDER BY score DESC
    LIMIT match_count;
END;
$$;
